"""Bereid modelinvoer voor een stroomgebied voor.

align: leg hoogte, bodembedekking en afstromingspercentage op hetzelfde raster.
terrain: bereken stroomrichting, bovenstrooms oppervlak en helling.
build: maak gebiedsgrenzen, verbindingen en celgegevens voor de modelapp.
Zie Cookbook_inputbestanden.pdf voor de werkwijze.
Installatie: python -m pip install numpy pandas geopandas rasterio shapely pyarrow whitebox
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from rasterio.transform import from_origin, rowcol, xy
from rasterio.warp import reproject, Resampling
from shapely.geometry import Point, shape
from shapely.ops import unary_union

GEO_FIELDS = ['D8slope', 'manning_n', 'runoff_ori_clipped', 'D8flowdirection',
              'D8flowaccumulation', 'breachedburnedDTM_5m', 'DTM_5m']
CATCHMENT_FIELDS = ['uitstroompunt_nummer', 'uitstroompunt_type', 'outlet_x',
                    'outlet_y', 'outlet_mtaw', 'K_s', 'Q_cap_ms']
DIRECTION_MAP = {1: (-1, 1), 2: (0, 1), 4: (1, 1), 8: (1, 0),
                 16: (1, -1), 32: (0, -1), 64: (-1, -1), 128: (-1, 0)}

NODATA = -9999.


def point_geometry(geom, label='Uitlaat'):
    if geom is not None and not geom.is_empty:
        if geom.geom_type == 'Point':
            return geom
        if geom.geom_type == 'MultiPoint' and len(geom.geoms) == 1:
            return geom.geoms[0]
    raise ValueError(f'{label}: verwacht één punt.')


def validate_inputs(rasters, labels, mask, cells, outlets, profile):
    """Controleer celwaarden, reservoirvelden en routes vóór het schrijven."""
    t = profile['transform']
    if not profile['crs'].is_projected or profile['crs'].linear_units != 'metre':
        raise ValueError('Gebruik een geprojecteerd CRS in meters.')
    if t.b != 0 or t.d != 0 or t.a <= 0 or t.e != -t.a:
        raise ValueError('Vierkante, noordgerichte rastercellen vereist.')
    rows, cols = np.where(mask)
    if len(set(rows)) < 2 or len(set(cols)) < 2:
        raise ValueError('Het modelgebied moet minstens twee rijen en kolommen bevatten.')
    for field, data in rasters.items():
        values = data[mask]
        if not np.isfinite(values).all() or np.isin(values, [-9999., -32768.]).any():
            raise ValueError(f'{field}: NoData/onbekende klasse binnen het modelgebied.')
    for field, invalid in [
        ('D8slope', rasters['D8slope'][mask] < 0),
        ('manning_n', rasters['manning_n'][mask] <= 0),
        ('runoff_ori_clipped', (rasters['runoff_ori_clipped'][mask] < 0) |
                              (rasters['runoff_ori_clipped'][mask] > 100)),
        ('D8flowaccumulation', rasters['D8flowaccumulation'][mask] <= 0),
    ]:
        if invalid.any():
            raise ValueError(f'{field}: buiten geldig bereik (S>=0, n>0, runoff 0–100%, A>0).')
    for row in outlets.itertuples():
        kind = str(row.uitstroompunt_type).strip()
        if kind not in ('Q', 'H', 'Geen'):
            raise ValueError('Uitstroompunt_type moet Q, H of Geen zijn.')
        if kind == 'Q':
            if not np.isfinite([row.K_s, row.Q_cap_ms]).all() or row.K_s <= 0 or row.Q_cap_ms < 0:
                raise ValueError('Voor Q: K_s moet positief zijn en Q_cap_ms minstens nul.')
        elif not np.isfinite(row.outlet_mtaw):
            raise ValueError('Voor H/Geen: vul outlet_mtaw in.')

    # Iedere route moet haar uitlaat bereiken; ook gebiedskoppelingen mogen geen lus vormen.
    fd = rasters['D8flowdirection']
    height, width = mask.shape
    downstream = np.full(mask.size, -1, dtype=np.int64)
    for r, c in zip(rows, cols):
        cid = int(labels[r, c])
        is_outlet = (r, c) == cells[cid]
        code = fd[r, c]
        if code == 0 and is_outlet:
            continue
        if code not in DIRECTION_MAP:
            raise ValueError(f'Ongeldige D8-code op rij {r}, kolom {c}.')
        dr, dc = DIRECTION_MAP[code]
        nr, nc = r+dr, c+dc
        inside = 0 <= nr < height and 0 <= nc < width and mask[nr, nc]
        if not is_outlet and (not inside or labels[nr, nc] != cid):
            raise ValueError(f'Gebied {cid}: route verlaat het gebied vóór de uitlaat.')
        if inside:
            downstream[r*width+c] = nr*width+nc
    degree = np.bincount(downstream[downstream >= 0], minlength=mask.size)
    valid = np.flatnonzero(mask)
    queue = deque(int(i) for i in valid if degree[i] == 0)
    count = 0
    while queue:
        i = queue.popleft()
        count += 1
        j = downstream[i]
        if j >= 0:
            degree[j] -= 1
            if degree[j] == 0:
                queue.append(int(j))
    if count != len(valid):
        raise ValueError('De afvoerroute bevat een lus; controleer de stroomrichting.')


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_raster(path):
    with rasterio.open(path) as src:
        if src.count != 1 or src.crs is None:
            raise ValueError(f'{path}: één band met CRS vereist.')
        return src.read(1, masked=True).astype(float).filled(np.nan), src.profile


def write_raster(path, data, profile):
    p = dict(profile, driver='GTiff', count=1, dtype='float64', nodata=NODATA,
             height=data.shape[0], width=data.shape[1], compress='deflate')
    with rasterio.open(path, 'w', **p) as dst:
        dst.write(np.where(np.isfinite(data), data, NODATA), 1)


def aligned(path, profile):
    data, other = read_raster(path)
    for field in ('crs', 'transform', 'width', 'height'):
        if other[field] != profile[field]:
            raise ValueError(f'{path}: afwijkend {field}; lijn eerst uit met align.')
    return data


def align(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    xmin, ymin, xmax, ymax = args.bounds
    L = args.resolution
    size = np.array([(xmax-xmin)/L, (ymax-ymin)/L])
    if L <= 0 or not np.isfinite(size).all() or (size < 2).any() or not np.allclose(size, np.rint(size)):
        raise ValueError('Bounds moeten een positief geheel aantal vierkante cellen omvatten.')
    profile = dict(crs=rasterio.crs.CRS.from_string(args.crs), transform=from_origin(xmin, ymax, L, L))
    if not profile['crs'].is_projected or profile['crs'].linear_units != 'metre':
        raise ValueError('Gebruik een geprojecteerd CRS in meters.')
    records = {}
    for name, path, method in [('DTM_5m', args.dtm, Resampling.bilinear),
                               ('landcover', args.landcover, Resampling.nearest),
                               ('runoff_pct', args.runoff, Resampling.average)]:
        if path.resolve() == (out/f'{name}.tif').resolve():
            raise ValueError('Bron en uitvoer mogen niet hetzelfde bestand zijn.')
        with rasterio.open(path) as src:
            if src.crs is None or src.count != 1:
                raise ValueError(f'{path}: één band en bron-CRS vereist.')
            data = np.full((int(round(size[1])), int(round(size[0]))), np.nan)
            reproject(source=src.read(1, masked=True).astype(float).filled(np.nan), destination=data,
                      src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan,
                      dst_transform=profile['transform'], dst_crs=profile['crs'],
                      dst_nodata=np.nan, resampling=method)
        write_raster(out/f'{name}.tif', data, profile)
        records[name] = dict(path=str(path.resolve()), sha256=digest(path), resampling=method.name)
    save_json(out/'alignment.json', dict(bounds=args.bounds, resolution=L, crs=args.crs, sources=records))


def whitebox_call(wbt, name, output, **kwargs):
    # Always write to a fresh temporary name: a stale output cannot pass a failed run.
    output = Path(output)
    with tempfile.TemporaryDirectory(prefix='wbt_') as td:
        temp = Path(td)/output.name
        messages = []
        code = getattr(wbt, name)(output=str(temp), callback=messages.append, **kwargs)
        if code != 0 or not temp.exists():
            raise RuntimeError(f'WhiteboxTools {name} mislukt: ' + '\n'.join(messages))
        output.write_bytes(temp.read_bytes())
    return dict(tool=name, arguments=kwargs, output=output.name, messages=messages)


def engine():
    from whitebox import WhiteboxTools
    wbt = WhiteboxTools()
    wbt.set_verbose_mode(False)
    return wbt


def terrain(source):
    source = source.resolve()
    z, profile = read_raster(source/'DTM_5m.tif')
    t = profile['transform']
    if not profile['crs'].is_projected or profile['crs'].linear_units != 'metre':
        raise ValueError('Routing vereist een geprojecteerd CRS met meters.')
    if t.b != 0 or t.d != 0 or t.a <= 0 or t.e != -t.a:
        raise ValueError('Vierkant noordgericht grid vereist.')
    wbt, calls = engine(), []
    routing = source/'breachedburnedDTM_5m.tif'
    fd_file = source/'D8flowdirection.tif'
    calls.append(whitebox_call(wbt, 'breach_depressions', routing,
                              dem=str(source/'DTM_5m.tif'), flat_increment=0.0001, fill_pits=True))
    calls.append(whitebox_call(wbt, 'd8_pointer', fd_file, dem=str(routing), esri_pntr=False))
    calls.append(whitebox_call(wbt, 'd8_flow_accumulation', source/'D8flowaccumulation.tif',
                              i=str(fd_file), out_type='catchment area', pntr=True,
                              esri_pntr=False, log=False, clip=False))
    fd, rz = aligned(fd_file, profile), aligned(routing, profile)
    slope = np.full(z.shape, np.nan)
    for r, c in zip(*np.where(np.isfinite(z))):
        if fd[r, c] == 0:
            slope[r, c] = 0.
            continue
        if fd[r, c] not in DIRECTION_MAP:
            raise ValueError(f'Ongeldige D8-code op {r}, {c}.')
        dr, dc = DIRECTION_MAP[fd[r, c]]
        nr, nc = r+dr, c+dc
        if not (0 <= nr < z.shape[0] and 0 <= nc < z.shape[1]) or not np.isfinite(rz[nr, nc]):
            raise ValueError('D8-route naar NoData; vergroot het brondomein.')
        value = (rz[r, c]-rz[nr, nc])/(t.a*np.hypot(dr, dc))
        if value < -1e-10:
            raise ValueError('D8-route loopt omhoog; richting en routingterrein komen niet overeen.')
        slope[r, c] = max(0., value)
    write_raster(source/'D8slope.tif', slope, profile)
    save_json(source/'terrain_processing.json', dict(whitebox_version=wbt.version().splitlines()[0],
               calls=calls, physical_sha256=digest(source/'DTM_5m.tif'),
               min_change_m=float(np.nanmin(rz-z)), max_change_m=float(np.nanmax(rz-z)),
               slope='max(0, (routing_z - downstream_z) / D8_distance_m); terminal=0'))


def build_inputs(source, output, name):
    source, output = source.resolve(), output.resolve()
    if not name or Path(name).name != name or any(c in name for c in '/\\:'):
        raise ValueError('Naam moet een enkele bestandsnaam zijn.')
    z, profile = read_raster(source/'DTM_5m.tif')
    rasters = {field: aligned(source/f'{field}.tif', profile) for field in GEO_FIELDS
               if field not in ('DTM_5m', 'manning_n', 'runoff_ori_clipped')}
    rasters['DTM_5m'] = z
    rasters['runoff_ori_clipped'] = aligned(source/'runoff_pct.tif', profile)
    cover = aligned(source/'landcover.tif', profile)
    table = pd.read_csv(source/'manning.csv')
    if not {'class_id', 'manning_n'}.issubset(table.columns):
        raise ValueError('manning.csv vereist class_id,manning_n.')
    if table.class_id.isna().any() or table.class_id.duplicated().any():
        raise ValueError('Manningtabel: ontbrekende of dubbele klassecodes.')
    manning = np.full(z.shape, np.nan)
    for row in table.itertuples():
        if not np.isfinite(row.manning_n) or row.manning_n <= 0:
            raise ValueError('Manning n moet positief zijn.')
        manning[cover == row.class_id] = row.manning_n
    rasters['manning_n'] = manning
    outlets = gpd.read_file(source/'outlets.gpkg')
    if outlets.empty:
        raise ValueError('Plaats minstens één uitlaat in outlets.gpkg.')
    required = [field for field in CATCHMENT_FIELDS if field not in ('outlet_x', 'outlet_y')]
    if not set(required).issubset(outlets.columns) or outlets.crs != profile['crs']:
        raise ValueError('Outlets: verplichte reservoirvelden ontbreken of CRS wijkt af.')
    ids = pd.to_numeric(outlets.uitstroompunt_nummer, errors='coerce').to_numpy(float)
    if not np.isfinite(ids).all() or (ids <= 0).any() or (ids > 2147483647).any() or (ids != np.floor(ids)).any() or len(set(ids)) != len(ids):
        raise ValueError('Uitlaat-ID’s moeten unieke positieve int32-getallen zijn.')
    seeds = np.zeros(z.shape)
    cells = {}
    for row in outlets.itertuples():
        p = point_geometry(row.geometry, 'Uitlaat')
        r, c = rowcol(profile['transform'], p.x, p.y)
        if not (0 <= r < z.shape[0] and 0 <= c < z.shape[1]) or not np.isfinite(z[r, c]):
            raise ValueError('Uitlaat buiten geldig terrein.')
        cx, cy = xy(profile['transform'], r, c)
        if np.hypot(p.x-cx, p.y-cy) > 1e-5:
            raise ValueError('Plaats iedere uitlaat exact op het gekozen celcentrum; geen automatische snap.')
        if seeds[r, c] != 0:
            raise ValueError('Twee uitlaten in dezelfde cel.')
        cid = int(row.uitstroompunt_nummer)
        seeds[r, c], cells[cid] = cid, (r, c)
    seedfile = source/'pour_ids.tif'
    write_raster(seedfile, np.where(np.isfinite(z), seeds, np.nan), profile)
    call = whitebox_call(engine(), 'watershed', source/'catchment_ids.tif',
                         d8_pntr=str(source/'D8flowdirection.tif'), pour_pts=str(seedfile), esri_pntr=False)
    labels = aligned(source/'catchment_ids.tif', profile)
    mask = np.isfinite(labels) & (labels > 0)
    if set(np.unique(labels[mask])) != set(cells):
        raise ValueError('Watershed-labels en uitlaat-ID’s verschillen.')
    validate_inputs(rasters, labels, mask, cells, outlets, profile)
    polygon_parts = {cid: [] for cid in cells}
    for geom, cid in shapes(np.where(mask, labels, 0).astype('int32'), mask=mask, transform=profile['transform']):
        polygon_parts[int(cid)].append(shape(geom))
    catchments, links, finals = [], [], []
    for row in outlets.itertuples():
        cid = int(row.uitstroompunt_nummer)
        r, c = cells[cid]
        record = {field: getattr(row, field) for field in required}
        record.update(outlet_x=float(row.geometry.centroid.x), outlet_y=float(row.geometry.centroid.y),
                      uitstroompunt_naam=getattr(row, 'uitstroompunt_naam', f'Gebied {cid}'),
                      geometry=unary_union(polygon_parts[cid]))
        catchments.append(record)
        dr, dc = DIRECTION_MAP.get(rasters['D8flowdirection'][r, c], (0, 0))
        nr, nc = r+dr, c+dc
        target = int(labels[nr, nc]) if (dr or dc) and 0 <= nr < z.shape[0] and 0 <= nc < z.shape[1] and mask[nr, nc] else None
        if target == cid:
            raise ValueError(f'Gebied {cid}: uitlaat leidt terug naar eigen gebied.')
        if target is None:
            finals.append(cid)
        else:
            links.append(dict(source_id=cid, target_id=target, geometry=Point(*xy(profile['transform'], nr, nc))))
    catchments = gpd.GeoDataFrame(catchments, crs=profile['crs'])
    inlets = gpd.GeoDataFrame(links, columns=['source_id', 'target_id', 'geometry'], crs=profile['crs'])
    for col in ('source_id', 'target_id'):
        inlets[col] = inlets[col].astype('int64')
    rows, cols = np.where(mask)
    xs, ys = xy(profile['transform'], rows, cols)
    geo = gpd.GeoDataFrame({key: val[mask].astype(float) for key, val in rasters.items()},
                           geometry=gpd.points_from_xy(xs, ys), crs=profile['crs'])
    crs, L = profile['crs'], profile['transform'].a
    output.mkdir(parents=True, exist_ok=True)
    geo.to_parquet(output/f'{name}_geodata.parquet', index=False)
    catchments.to_file(output/'catchments.gpkg', layer='catchments', driver='GPKG', mode='w')
    inlets.to_file(output/'inlet_pts.gpkg', layer='inlet_pts', driver='GPKG', mode='w')
    report = dict(status='passed', cell_count=len(geo), area_m2=len(geo)*L**2, crs=str(crs), resolution_m=L,
                  catchments={cid: int((labels == cid).sum()) for cid in cells}, external_outlets=finals,
                  links=inlets.drop(columns='geometry').to_dict('records'))
    save_json(output/'validation.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'cell_count', 'area_m2', 'catchments', 'links', 'external_outlets')}, indent=2))



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('align', help='Leg de drie bronkaarten op hetzelfde raster.')
    for name in ('dtm', 'landcover', 'runoff', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--bounds', type=float, nargs=4, required=True, metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'))
    p.add_argument('--crs', default='EPSG:31370')
    p.add_argument('--resolution', type=float, default=5.)
    p = commands.add_parser('terrain', help='Bereken stroomrichting, bovenstrooms oppervlak en helling.')
    p.add_argument('--source', type=Path, required=True)
    p = commands.add_parser('build', help='Maak en controleer de drie modelinvoerbestanden.')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--name', default='gebied')
    args = parser.parse_args()
    if args.command == 'align':
        align(args)
    elif args.command == 'terrain':
        terrain(args.source)
    elif args.command == 'build':
        build_inputs(args.source, args.output, args.name)


if __name__ == '__main__':
    main()
