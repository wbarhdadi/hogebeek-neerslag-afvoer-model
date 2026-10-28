"""Input validation and raster preparation for the existing three-file contract."""
from collections import deque
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.transform import from_origin
from routing import D8Topology, DIRECTION_MAP

GEO_FIELDS = ['D8slope', 'manning_n', 'runoff_ori_clipped', 'D8flowdirection',
              'D8flowaccumulation', 'breachedburnedDTM_5m', 'DTM_5m']
CATCHMENT_FIELDS = ['uitstroompunt_nummer', 'uitstroompunt_type', 'outlet_x',
                    'outlet_y', 'outlet_mtaw', 'K_s', 'Q_cap_ms']
NODATA = (-9999., -32768.)


def _fields(frame, names, label):
    missing = sorted(set(names)-set(frame.columns))
    if missing:
        raise ValueError(f'{label}: verplichte velden ontbreken: {missing}.')


def _ids(values, label):
    vals = pd.to_numeric(values, errors='coerce').to_numpy(float)
    if not np.isfinite(vals).all() or (vals <= 0).any() or (vals != np.floor(vals)).any():
        raise ValueError(f'{label}: gebieds-ID moet een positief geheel getal zijn.')
    return vals.astype(int)


def point_geometry(geom, label='Inlaat'):
    if geom is None or geom.is_empty:
        raise ValueError(f'{label}: lege geometrie.')
    if geom.geom_type == 'Point':
        return geom
    if geom.geom_type == 'MultiPoint' and len(geom.geoms) == 1:
        return geom.geoms[0]
    raise ValueError(f'{label}: verwacht Point of MultiPoint met één punt; meerdere punten vereisen een expliciete verdeling van het volume.')


def validate_network(catchments, inlets):
    _fields(catchments, CATCHMENT_FIELDS, 'Catchments')
    _fields(inlets, ['source_id', 'target_id'], 'Inlets')
    ids = _ids(catchments.uitstroompunt_nummer, 'Catchments')
    if len(ids) == 0 or len(set(ids)) != len(ids):
        raise ValueError('Catchments: lege tabel of dubbele gebieds-ID’s.')
    source, target = _ids(inlets.source_id, 'Inlets source_id'), _ids(inlets.target_id, 'Inlets target_id')
    if len(source) != len(set(source)):
        raise ValueError('Inlets: een brongebied heeft meerdere verbindingen; volumeverdeling ontbreekt.')
    if not set(source).union(target).issubset(ids):
        raise ValueError('Inlets: onbekend source_id of target_id.')
    graph = dict(zip(source, target))
    degree = {int(cid): 0 for cid in ids}
    for t in target:
        degree[t] += 1
    queue = deque(cid for cid in ids if degree[cid] == 0)
    order = []
    while queue:
        s = queue.popleft()
        order.append(int(s))
        if s in graph:
            t = graph[s]
            degree[t] -= 1
            if degree[t] == 0:
                queue.append(t)
    if len(order) != len(ids):
        raise ValueError('Netwerk bevat een cyclus; run geblokkeerd.')
    for row in catchments.itertuples():
        kind, cid = str(row.uitstroompunt_type).strip(), row.uitstroompunt_nummer
        if kind not in ('Q', 'H', 'Geen'):
            raise ValueError(f'Gebied {cid}: onopgelost reservoirtype {kind!r}; run geblokkeerd.')
        if not np.isfinite([row.outlet_x, row.outlet_y]).all():
            raise ValueError(f'Gebied {cid}: uitlaatcoördinaten ontbreken.')
        if kind == 'Q':
            if not np.isfinite([row.K_s, row.Q_cap_ms]).all() or row.K_s <= 0 or row.Q_cap_ms < 0:
                raise ValueError(f'Gebied {cid}: ongeldige K_s of Q_cap_ms.')
        elif not np.isfinite(row.outlet_mtaw):
            raise ValueError(f'Gebied {cid}: H-drempel outlet_mtaw ontbreekt.')
    return order


def _grid_size(x, y):
    dx, dy = np.diff(np.sort(np.unique(x))), np.diff(np.sort(np.unique(y)))
    if not len(dx) or not len(dy):
        raise ValueError('Rasterresolutie kan niet uit minder dan twee rijen/kolommen worden afgeleid.')
    L = float(min(dx.min(), dy.min()))
    if L <= 0 or not np.allclose(dx/L, np.round(dx/L), rtol=0, atol=1e-6) or not np.allclose(dy/L, np.round(dy/L), rtol=0, atol=1e-6):
        raise ValueError('Geodata: onregelmatig of niet-vierkant rastergrid.')
    # At least one observed adjacent pair in each dimension is required.
    if not np.isclose(dx.min(), L) or not np.isclose(dy.min(), L):
        raise ValueError('Geodata: rastercellen zijn niet vierkant.')
    return L


def coordinate_to_index(transform, shape, mask, px, py, label):
    col_float, row_float = (~transform)*(px, py)
    row, col = int(np.floor(row_float)), int(np.floor(col_float))
    if not (0 <= row < shape[0] and 0 <= col < shape[1]) or not mask[row, col]:
        raise ValueError(f'{label} ({px:.3f}, {py:.3f}) ligt niet op een geldige gebiedscel; geen automatische verplaatsing.')
    return row, col


def outlet_to_index(transform, mask, fd, px, py, label):
    """Resolve a point to its unique D8 terminal within half a cell diagonal.

    Official outlet 4 lies just outside its cell footprint, but 3.467 m from
    the unique terminal centre on a 5 m grid. The D8 topology provides the
    evidence; never snap to an arbitrary nearby/upstream cell or remote outlet.
    """
    terminals = []
    height, width = mask.shape
    for r, c in zip(*np.where(mask)):
        dr, dc = DIRECTION_MAP.get(fd[r, c], (0, 0))
        nr, nc = r+dr, c+dc
        if fd[r, c] == 0 or (fd[r, c] in DIRECTION_MAP and
                            (not (0 <= nr < height and 0 <= nc < width) or not mask[nr, nc])):
            terminals.append((int(r), int(c)))
    if len(terminals) != 1:
        raise ValueError(f'{label}: verwacht één D8-eindcel, gevonden {len(terminals)}.')
    rc = terminals[0]
    x, y = transform*(rc[1]+.5, rc[0]+.5)
    distance = float(np.hypot(x-px, y-py))
    tolerance = float(np.hypot(transform.a, transform.e)/2)
    if distance > tolerance+1e-6:
        raise ValueError(f'{label}: D8-eindcel ligt {distance:.3f} m van opgegeven punt; maximaal {tolerance:.3f} m toegestaan.')
    return rc, dict(source_x=float(px), source_y=float(py), cell_x=x, cell_y=y,
                    distance_m=distance, max_distance_m=tolerance, method='unique D8 terminal within half cell diagonal')


def preprocess_geodata(geodata, catchments, inlets, maatregelen=None, slope_factor=1.,
                       manning_factor=1., runoff_factor=1., apply_measures=None, log=print):
    _fields(geodata, GEO_FIELDS, 'Geodata')
    validate_network(catchments, inlets)
    catchments, inlets = catchments.copy(), inlets.copy()
    catchments['uitstroompunt_nummer'] = _ids(catchments.uitstroompunt_nummer, 'Catchments')
    inlets['source_id'] = _ids(inlets.source_id, 'Inlets source_id')
    inlets['target_id'] = _ids(inlets.target_id, 'Inlets target_id')
    for label, frame in [('Geodata', geodata), ('Catchments', catchments), ('Inlets', inlets)]:
        if frame.crs is None:
            raise ValueError(f'{label}: CRS ontbreekt.')
        if frame.geometry.isna().any() or frame.geometry.is_empty.any() or not frame.geometry.is_valid.all():
            raise ValueError(f'{label}: lege of ongeldige geometrie.')
    if not geodata.crs.is_projected or any(a.unit_conversion_factor != 1 for a in geodata.crs.axis_info):
        raise ValueError('Geodata: een geprojecteerd CRS met meters is vereist.')
    if geodata.crs != catchments.crs:
        raise ValueError('CRS geodata en catchments verschilt; reprojiceer rasters eerst naar één uitgelijnd grid.')
    if not geodata.geom_type.eq('Point').all():
        raise ValueError('Geodata moet uit Point-celcentra bestaan.')
    if not catchments.geom_type.isin(['Polygon', 'MultiPolygon']).all():
        raise ValueError('Catchments moeten Polygon/MultiPolygon zijn.')
    polys = list(catchments.geometry)
    for i, a in enumerate(polys):
        for b in polys[i+1:]:
            if a.intersection(b).area > 1e-6:
                raise ValueError('Catchmentpolygonen overlappen; gebruik lokale, niet-cumulatieve gebieden.')
    inlets = inlets.to_crs(geodata.crs)
    for row in inlets.itertuples():
        point_geometry(row.geometry, f'Inlaat {row.source_id} → {row.target_id}')
    factors = np.array([slope_factor, manning_factor, runoff_factor], float)
    if not np.isfinite(factors).all() or slope_factor <= 0 or manning_factor <= 0 or runoff_factor < 0:
        raise ValueError('Helling-/Manningfactor moeten positief zijn; runofffactor moet niet-negatief zijn.')
    data = geodata.copy()
    data['x'], data['y'] = data.geometry.x, data.geometry.y
    if data[['x', 'y']].duplicated().any():
        raise ValueError('Geodata: dubbele celcentra.')
    L = _grid_size(data.x, data.y)
    # Boundary centres belong to a polygon too; multiple assignments are errors.
    joined = gpd.sjoin(data, catchments[['uitstroompunt_nummer', 'geometry']], how='inner', predicate='intersects')
    if joined.index.duplicated().any():
        raise ValueError('Een rastercel valt in meerdere catchments.')
    joined = joined.drop(columns='index_right').rename(columns={'uitstroompunt_nummer': 'catchment_id'})
    log(f'GIS: {len(joined)} gebiedscellen, {len(data)-len(joined)} cellen buiten het modeldomein; grid {L:g} m.')
    if maatregelen is not None:
        if apply_measures is None:
            raise ValueError('Maatregelenfunctie ontbreekt.')
        if maatregelen.crs is None:
            raise ValueError('Maatregelen: CRS ontbreekt.')
        joined = apply_measures(joined.copy(), maatregelen.to_crs(data.crs), catchments, log=log)
    inputs = {}
    for row in catchments.itertuples():
        cid = int(row.uitstroompunt_nummer)
        frame = joined[joined.catchment_id == cid]
        if frame.empty:
            raise ValueError(f'Gebied {cid}: geen rastercellen.')
        values = frame[GEO_FIELDS].apply(pd.to_numeric, errors='coerce').to_numpy(float)
        bad = ~np.isfinite(values) | np.isin(values, NODATA)
        if bad.any():
            ri, ci = np.argwhere(bad)[0]
            raise ValueError(f'Gebied {cid}: NoData/ongeldige waarde in {GEO_FIELDS[ci]} op bronrij {frame.index[ri]}.')
        x0, y0 = frame.x.min(), frame.y.max()
        cols = np.rint((frame.x.to_numpy()-x0)/L).astype(int)
        rows = np.rint((y0-frame.y.to_numpy())/L).astype(int)
        shape = (int(rows.max()+1), int(cols.max()+1))
        mask = np.zeros(shape, bool)
        mask[rows, cols] = True
        arrays = {}
        for name in GEO_FIELDS:
            arr = np.full(shape, np.nan)
            arr[rows, cols] = frame[name].to_numpy(float)
            arrays[name] = arr
        arrays['D8slope'] *= slope_factor
        arrays['manning_n'] *= manning_factor
        arrays['runoff_ori_clipped'] *= runoff_factor
        for name, invalid in [('D8slope', arrays['D8slope'][mask] < 0),
                              ('manning_n', arrays['manning_n'][mask] <= 0),
                              ('runoff_ori_clipped', (arrays['runoff_ori_clipped'][mask] < 0) | (arrays['runoff_ori_clipped'][mask] > 100)),
                              ('D8flowaccumulation', arrays['D8flowaccumulation'][mask] <= 0)]:
            if invalid.any():
                raise ValueError(f'Gebied {cid}: {name} buiten geldig bereik (runoff 0–100%, n>0, S>=0, A>0).')
        transform = from_origin(x0-L/2, y0+L/2, L, L)
        outlet, outlet_mapping = outlet_to_index(transform, mask, arrays['D8flowdirection'], row.outlet_x, row.outlet_y, f'Uitlaat {cid}')
        inlet_dict = {}
        for inlet in inlets[inlets.target_id == cid].itertuples():
            p = point_geometry(inlet.geometry)
            inlet_dict[int(inlet.source_id)] = coordinate_to_index(transform, shape, mask, p.x, p.y, f'Inlaat {inlet.source_id} → {cid}')
        try:
            topology = D8Topology.build(arrays['D8flowdirection'], outlet, mask)
        except ValueError as exc:
            raise ValueError(f'Gebied {cid}: {exc}') from exc
        inputs[cid] = dict(S_catchment=arrays['D8slope'], fd_catchment=arrays['D8flowdirection'],
                           fa_catchment=arrays['D8flowaccumulation'], ro_catchment=arrays['runoff_ori_clipped'],
                           manning_catchment=arrays['manning_n'], dtm_catchment=arrays['DTM_5m'],
                           routing_dtm_catchment=arrays['breachedburnedDTM_5m'],
                           outlet_indices=outlet, inlet_dict=inlet_dict, transform=transform, topology=topology,
                           outlet_mapping=outlet_mapping)
    return inputs, geodata.crs, L
