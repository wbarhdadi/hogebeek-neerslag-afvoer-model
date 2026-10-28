"""Small Leaflet viewer for the maximum-depth GeoTIFF in a completed run."""
import base64
from io import BytesIO
import json
import zipfile

import numpy as np
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
import streamlit as st


# Shared, fixed classes keep the colours comparable between areas and runs.
BREAKS = [.1, .25, .5, 1., 2.]
COLORS = ['#c6e9ff', '#83c9ee', '#4298d0', '#2166ac', '#253494', '#160b54']
LABELS = ['0–0,10', '0,10–0,25', '0,25–0,50', '0,50–1,00', '1,00–2,00', '> 2,00']


def depth_image(tiff):
    """Warp to the map's Mercator grid before placing the PNG by its bounds."""
    with MemoryFile(tiff) as memory, memory.open() as source:
        grid, width, height = calculate_default_transform(
            source.crs, 'EPSG:3857', source.width, source.height, *source.bounds)
        # Bound browser payload size; the downloadable GeoTIFF stays untouched.
        scale = max(1., width / 1600, height / 1600)
        out_width, out_height = max(1, int(width / scale)), max(1, int(height / scale))
        grid = grid * grid.scale(width / out_width, height / out_height)
        original = source.read(1, masked=True)
        maximum = float(original.max()) if original.count() else 0.
        depth = np.full((out_height, out_width), np.nan, dtype='float32')
        reproject(
            source=original.filled(np.nan), destination=depth,
            src_transform=source.transform, src_crs=source.crs, src_nodata=np.nan,
            dst_transform=grid, dst_crs='EPSG:3857', dst_nodata=np.nan,
            resampling=Resampling.nearest)
    palette = np.array([[int(color[i:i+2], 16) for i in (1, 3, 5)] for color in COLORS], dtype='uint8')
    rgba = np.zeros((*depth.shape, 4), dtype='uint8')
    rgba[:, :, :3] = palette[np.digitize(depth, BREAKS, right=True)]
    rgba[:, :, 3] = np.where(np.isfinite(depth) & (depth > 0), 255, 0)
    image = BytesIO()
    Image.fromarray(rgba).save(image, format='PNG')
    encoded = base64.b64encode(image.getvalue()).decode('ascii')
    west, south, east, north = transform_bounds(
        'EPSG:3857', 'EPSG:4326', *array_bounds(out_height, out_width, grid))
    return dict(image='data:image/png;base64,' + encoded,
                bounds=[[south, west], [north, east]], maximum=maximum)


@st.cache_data(max_entries=12, show_spinner=False)
def result_map_html(zip_bytes, cid, area):
    with zipfile.ZipFile(BytesIO(zip_bytes)) as archive:
        payload = depth_image(archive.read(f'max_waterdepth/max_waterdepth_catchment_{cid:02d}.tif'))
    payload.update(area, colors=COLORS, labels=LABELS)
    # Keep uploaded geometry/data inside the JSON literal, never executable HTML.
    data = json.dumps(payload, allow_nan=False).replace('<', '\\u003c')
    return _HTML.replace('__MAP_DATA__', data)


_HTML = '''<!doctype html>
<html lang="nl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
html, body, #map { height:100%; margin:0; font-family:Arial,sans-serif; }
.panel { background:white; color:#222; padding:10px; border-radius:5px; box-shadow:0 1px 5px #777; }
.legend-row { display:flex; align-items:center; gap:7px; margin-top:4px; }
.swatch { width:22px; height:13px; }
.panel label { display:block; margin-top:8px; }
.panel input { display:block; width:150px; }
.leaflet-image-layer { image-rendering:pixelated; }
#status { position:absolute; top:8px; left:55px; z-index:1000; background:white; padding:6px; }
#status:empty { display:none; }
</style></head><body>
<div id="map" aria-label="Kaart maximale waterdiepte"></div>
<div id="status">Kaart laden… Hiervoor is internet nodig.</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const data = __MAP_DATA__;
if (typeof L === 'undefined') {
    document.getElementById('status').textContent = 'Kaartbibliotheek niet bereikbaar. Controleer de internetverbinding.';
} else {
    const status = document.getElementById('status');
    status.textContent = '';
    const map = L.map('map', {maxZoom:22, scrollWheelZoom:false}).fitBounds(data.bounds);
    const attribution = '&copy; <a href="https://www.vlaanderen.be/digitaal-vlaanderen" target="_blank" rel="noopener">Digitaal Vlaanderen</a>';
    const ortho = L.tileLayer.wms('https://geo.api.vlaanderen.be/OMWRGBMRVL/wms', {
        layers:'Ortho', format:'image/jpeg', version:'1.1.1', maxZoom:22, attribution
    }).addTo(map);
    const grb = L.tileLayer.wms('https://geo.api.vlaanderen.be/GRB/wms', {
        layers:'GRB_ADP_GRENS,GRB_GBG,GRB_WLAS', format:'image/png', transparent:true,
        version:'1.1.1', opacity:0.5, maxZoom:22, attribution
    });
    for (const layer of [ortho, grb]) {
        layer.on('tileerror', () => { status.textContent = 'Een achtergrondlaag kon niet laden. Controleer de internetverbinding of probeer later opnieuw.'; });
    }
    const depth = L.imageOverlay(data.image, data.bounds, {opacity:0.7}).addTo(map);
    const boundary = L.geoJSON(data.boundary, {
        style:{color:'#ffde59', weight:3, fill:false}
    }).addTo(map);
    const outlet = L.circleMarker(data.outlet, {
        radius:7, color:'#fff', weight:2, fillColor:'#e53935', fillOpacity:1
    }).bindTooltip('Uitlaatpunt').addTo(map);
    map.fitBounds(boundary.getBounds().pad(0.08));
    L.control.layers({'Orthofoto meest recent':ortho}, {
        'GRB (percelen, gebouwen, waterlopen)':grb,
        'Maximale waterdiepte':depth, 'Stroomgebied':boundary, 'Uitlaatpunt':outlet
    }, {collapsed:false}).addTo(map);
    L.control.scale({imperial:false}).addTo(map);
    const legend = L.control({position:'bottomright'});
    legend.onAdd = () => {
        const panel = L.DomUtil.create('div', 'panel');
        panel.innerHTML = '<strong>Maximale waterdiepte [m]</strong>' +
            data.colors.map((color, i) => '<div class="legend-row"><span class="swatch" style="background:' + color + '"></span>' + data.labels[i] + '</div>').join('') +
            '<div style="margin-top:6px">Droog / geen data: transparant</div>' +
            '<div>Maximum: ' + data.maximum.toLocaleString('nl-BE', {maximumFractionDigits:3}) + ' m</div>' +
            (data.maximum <= 0 ? '<div>Geen water op maaiveld.</div>' : '') +
            '<label>Ondoorzichtigheid waterdiepte<input aria-label="Ondoorzichtigheid waterdiepte" id="depth-opacity" type="range" min="0" max="1" step="0.05" value="0.7"></label>' +
            '<label>Ondoorzichtigheid GRB<input aria-label="Ondoorzichtigheid GRB" id="grb-opacity" type="range" min="0" max="1" step="0.05" value="0.5"></label>';
        L.DomEvent.disableClickPropagation(panel);
        L.DomEvent.disableScrollPropagation(panel);
        panel.querySelector('#depth-opacity').oninput = event => depth.setOpacity(Number(event.target.value));
        panel.querySelector('#grb-opacity').oninput = event => grb.setOpacity(Number(event.target.value));
        return panel;
    };
    legend.addTo(map);
}
</script></body></html>'''
