"""File handling and immutable result exports for the single-page app."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.warp import transform
from shapely.geometry import mapping


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def file_identity(upload):
    if upload is None:
        return None
    data = upload.getbuffer()
    return dict(name=Path(upload.name).name, size_bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest())


def code_identity():
    base = Path(__file__).resolve().parent
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(base.glob('*.py'))}
    files['requirements.txt'] = hashlib.sha256((base/'requirements.txt').read_bytes()).hexdigest()
    return dict(sha256=fingerprint(files), files=files)


def rain_identity(frame):
    return fingerprint(dict(attrs=frame.attrs,
                            csv_sha256=hashlib.sha256(frame.to_csv(index=False).encode('utf-8')).hexdigest()))


def read_inputs(geodata_file, catchments_file, inlets_file, measures_files):
    """Read fresh input copies per run; never reuse mutated measure inputs."""
    geodata = gpd.read_parquet(BytesIO(geodata_file.getvalue()))
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        vectors = []
        for filename, upload in [('catchments.gpkg', catchments_file), ('inlet_pts.gpkg', inlets_file)]:
            path = root/filename
            path.write_bytes(upload.getvalue())
            vectors.append(gpd.read_file(path))
        measures = None
        if measures_files:
            folder = root/'measures'
            folder.mkdir()
            names = set()
            for upload in measures_files:
                name = Path(upload.name).name
                if name.lower() in names:
                    raise ValueError(f'Maatregelen: dubbele bestandsnaam {name}; geef iedere shapefile een unieke naam.')
                names.add(name.lower())
                (folder/name).write_bytes(upload.getvalue())
            shapes = sorted(p for p in folder.iterdir() if p.suffix.lower() == '.shp')
            if not shapes:
                raise ValueError('Maatregelen: upload ook het .shp-bestand en bijbehorende .dbf, .shx en .prj.')
            frames = [gpd.read_file(p) for p in shapes]
            if any(f.crs is None for f in frames):
                raise ValueError('Maatregelen: CRS ontbreekt; voeg de bijbehorende .prj toe.')
            measures = gpd.GeoDataFrame(pd.concat([f.to_crs(geodata.crs) for f in frames],
                                                 ignore_index=True), crs=geodata.crs)
            if 'type' not in measures or measures['type'].isna().any():
                raise ValueError('Maatregelen: vul het verplichte veld type in voor iedere maatregel.')
    return geodata, *vectors, measures


def catchment_names(catchments):
    names = {}
    for row in catchments.itertuples():
        cid = int(row.uitstroompunt_nummer)
        name = getattr(row, 'uitstroompunt_naam', None)
        names[cid] = f'{name} (gebied {cid})' if pd.notna(name) and str(name).strip() else f'Gebied {cid}'
    return names


def prepare_results(result, rain, reservoirs, inputs, catchments, config, signature, logs):
    """Build once after success; chart selection/download never reruns the model."""
    names = catchment_names(catchments)
    ids = sorted(names)
    q = pd.DataFrame({'datum': result.times, **{f'debiet_c{c}_m3_s': result.discharges[c] for c in ids}})
    v = pd.DataFrame({'datum': result.times, **{f'volume_c{c}_m3_step': result.volumes[c] for c in ids}})
    h = pd.DataFrame({'datum': result.times, **{f'waterpeil_c{c}_m_DTM_referentie': result.waterlevels[c] for c in ids}})
    # Maximum spatial depth at each interval end, consistent with the exported
    # depth raster and the physical (possibly modified) terrain used for storage.
    depths = {c: np.maximum(0., result.waterlevels[c]-np.nanmin(reservoirs[c].dtm)) for c in ids}
    metadata = dict(result.metadata, schema_version=2, created_at_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                    run_signature=signature, configuration=config, catchment_names=names,
                    horizontal_crs=str(reservoirs[ids[0]].crs),
                    vertical_reference='DTM vertical reference; not verified as TAW',
                    balance=result.balance, balance_per_catchment=result.balance_per_catchment,
                    outlet_mapping={c: inputs[c]['outlet_mapping'] for c in ids},
                    measures=dict(files=config['inputs']['measures'],
                                  application_passes=int(bool(config['inputs']['measures'])),
                                  processing_log=[s for s in logs if 'maatregel' in s.lower()]))
    metadata['units'] = dict(metadata['units'], rainfall='mm/step', time='UTC', depth='m above DTM_5m')
    output = BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for filename, frame in [('catchment_discharges.csv', q), ('catchment_volumes.csv', v),
                                ('catchment_waterlevels.csv', h), ('rainfall_mm_step.csv', rain)]:
            archive.writestr(filename, frame.to_csv(index=False))
        archive.writestr('run_metadata.json', json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False))
        archive.writestr('run_log.txt', '\n'.join(logs))
        with tempfile.TemporaryDirectory() as directory:
            for cid in ids:
                highest = max(float(np.max(result.waterlevels[cid])), result.metadata['initial_levels_m'][cid])
                path = Path(directory)/f'max_waterdepth_catchment_{cid:02d}.tif'
                reservoirs[cid].export_waterdepth(path, highest)
                archive.write(path, f'max_waterdepth/{path.name}')
    boundaries = catchments.to_crs('EPSG:4326')
    map_areas = {}
    for row in boundaries.itertuples():
        cid = int(row.uitstroompunt_nummer)
        outlet = inputs[cid]['outlet_mapping']
        lon, lat = transform(reservoirs[cid].crs, 'EPSG:4326',
                             [outlet['source_x']], [outlet['source_y']])
        map_areas[cid] = dict(boundary=mapping(row.geometry), outlet=[lat[0], lon[0]])
    return dict(signature=signature, result=result, rain=rain.copy(deep=True), names=names,
                map_areas=map_areas,
                depths=depths, metadata=metadata, logs=list(logs), zip_bytes=output.getvalue(),
                filename=f'hogebeek_resultaten_{signature[:12]}.zip')
