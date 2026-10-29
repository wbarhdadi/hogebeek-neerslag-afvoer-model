"""Hoge Beek: Invoer → Bui → Berekenen → Resultaten."""
from pathlib import Path

import pandas as pd
import streamlit as st

from app_support import (code_identity, file_identity, fingerprint, prepare_results,
                         rain_identity, read_inputs)
from charts import chart_spec, local_time, result_charts, time_axis
from comparison import SCENARIOS, comparison_table, matching_runs, scenario, style_comparison_table
from gis import preprocess_geodata
from measures import apply_maatregelen
from model import run_model
from rainfall import fetch_waterinfo, read_rain_csv, scale_rain
from reservoirs import build_reservoirs
from result_map import result_map_html
from stations import STATIONS


def waterinfo_window(start_text, end_text):
    try:
        start, end = pd.Timestamp(start_text), pd.Timestamp(end_text)
        start = start.tz_localize('Europe/Brussels') if start.tzinfo is None else start
        end = end.tz_localize('Europe/Brussels') if end.tzinfo is None else end
    except Exception as exc:
        raise ValueError('Ongeldige of dubbelzinnige datum. Gebruik JJJJ-MM-DD of een tijdstip met UTC-offset.') from exc
    if pd.isna(start) or pd.isna(end) or not pd.Timedelta(0) < end-start <= pd.Timedelta(hours=72):
        raise ValueError('Kies een regenperiode van meer dan 0 en maximaal 72 uur; er wordt niets afgekapt.')
    return start, end


st.set_page_config(page_title='Hoge Beek – neerslag en afvoer', layout='wide')
for key, initial in [('prepared_rain', None), ('completed_run', None), ('run_error', None),
                     ('logs', []), ('run_history', {})]:
    st.session_state.setdefault(key, initial)
# Preserve an existing successful run when this feature is loaded in a live session.
if st.session_state.completed_run is not None and not st.session_state.run_history:
    previous = st.session_state.completed_run
    st.session_state.run_history = {scenario(previous['metadata']['configuration']): previous}

st.title('Hoge Beek – neerslag en afvoer')
st.caption('Invoer → Bui → Berekenen → Resultaten')

st.header('1. Invoer')
st.write('Upload de drie gebiedsbestanden. De bestandsnamen mogen afwijken.')
cols = st.columns(3)
with cols[0]:
    geodata_file = st.file_uploader('Geodata (.parquet)', type=['parquet'], key='geodata')
with cols[1]:
    catchments_file = st.file_uploader('Catchments (.gpkg)', type=['gpkg'], key='catchments')
with cols[2]:
    inlets_file = st.file_uploader('Inlets (.gpkg)', type=['gpkg'], key='inlets')

measures_files = st.file_uploader('Maatregelen (alle shapefile-onderdelen, optioneel)',
                                  type=['shp', 'dbf', 'shx', 'prj', 'cpg', 'qix'],
                                  accept_multiple_files=True, key='measures')
st.caption('Bereken eerst zonder maatregelen, upload daarna de maatregelen en bereken opnieuw. '
           'Houd alle overige invoer en instellingen gelijk om de resultaten te vergelijken.')
st.caption('Veld type: drempel (punt); gracht, houthaksel, houtkant, infiltratiestraat, swale (lijn); '
           'bufferbekken, grasbufferstrook, ontharding, sponsakker, sponstuin (polygoon). '
           'Upload per shapefile ook .dbf, .shx en .prj.')

with st.expander('Specialistische parameters'):
    st.caption('Deze factoren wijzigen de gebiedseigenschappen. De neerslagfactor staat apart bij Bui.')
    cols = st.columns(3)
    with cols[0]:
        slope_factor = st.number_input('Factor helling', min_value=0.001, value=1.0, key='slope_factor')
    with cols[1]:
        manning_factor = st.number_input('Factor Manning n', min_value=0.001, value=1.0, key='manning_factor')
    with cols[2]:
        runoff_factor = st.number_input('Runofffactor', min_value=0.0, value=1.0, key='runoff_factor',
                                        help='Vermenigvuldigt het runoffpercentage. Waarden boven 100% blokkeren de run.')
    A_threshold = st.number_input('Kanaaldrempel stroomopwaarts oppervlak [m²]',
                                  min_value=0.0, value=100000.0, key='threshold')
    tail_hours = st.number_input('Droge uitloop [uren]', min_value=0, max_value=720, value=72, key='tail_hours')
    timestamp_position = st.selectbox('Waterinfo-timestamp is intervalbegin of -einde', ['start', 'end'],
                                      key='timestamp_position',
                                      help='Historische bronconventie nog te bevestigen; start is de bestaande modelaanname.')

st.header('2. Bui')
source = st.selectbox('Neerslagbron', ['Waterinfo', 'Eigen CSV'], key='rain_source')
cols = st.columns(2)
with cols[0]:
    timestep_minutes = st.number_input('Modeltijdstap [minuten]', min_value=5, max_value=180,
                                       value=60, step=5, key='dt_minutes')
with cols[1]:
    precipitation_factor = st.number_input('Neerslagfactor', min_value=0.0, value=1.0, step=0.1,
                                            key='precipitation_factor',
                                            help='Eenmaal toegepast na conversie naar mm per modelstap; 0 = droog, 2 = dubbele regen.')
dt_seconds = int(timestep_minutes)*60
csv_file = None
fetch_clicked = False
if source == 'Waterinfo':
    stations = dict(STATIONS)
    station_id = st.selectbox('Station', list(stations), index=list(stations).index(61016042),
                              format_func=lambda sid: f'{stations[sid]} ({sid})', key='station_id')
    cols = st.columns(2)
    with cols[0]:
        start_text = st.text_input('Begin (inclusief)', '2025-10-03', key='rain_start')
    with cols[1]:
        end_text = st.text_input('Einde (exclusief)', '2025-10-06', key='rain_end')
    st.caption('Datums zonder tijdzone gelden in Europe/Brussels. Een tijdstip met offset mag ook, '
               'bijvoorbeeld 2025-10-03T12:00:00+02:00. Maximaal 72 uur regen.')
    rain_settings = dict(source=source, ts_id=str(station_id), station_name=stations[station_id],
                         start=start_text, end=end_text, timestamp_position=timestamp_position, dt_s=dt_seconds)
    fetch_clicked = st.button('Neerslag ophalen / vernieuwen', key='fetch_rain')
else:
    st.write('CSV: `timestamp,precipitation_mm`. Hoeveelheid in mm per interval; timestamp aan '
             'intervalbegin met tijdzone-offset. Neem ook droge intervallen op, met waarde 0.')
    csv_interval = st.number_input('Vaste CSV-intervalduur [minuten]', min_value=1, max_value=1440,
                                   value=15, key='csv_interval')
    csv_file = st.file_uploader('Eigen neerslag (.csv)', type=['csv'], key='rain_csv')
    st.download_button('Download voorbeeld-CSV (15 minuten)',
                       data=(Path(__file__).parent/'examples/neerslag_15min.csv').read_bytes(),
                       file_name='neerslag_15min.csv', mime='text/csv', on_click='ignore', key='csv_example')
    st.caption('Het voorbeeld bevat 6 mm in één uur. Het volledige bestand wordt gebruikt (maximaal 72 uur); '
               'een onvolledige modelstap of ontbrekend interval geeft een foutmelding.')
    rain_settings = dict(source=source, csv=file_identity(csv_file), interval_seconds=int(csv_interval)*60,
                         timestamp_position='start', dt_s=dt_seconds)

rain_key = fingerprint(rain_settings)
prepared = st.session_state.prepared_rain
rain = None
try:
    if fetch_clicked:
        # A failed refresh must not make a previous download appear refreshed.
        st.session_state.prepared_rain = prepared = None
        start, end = waterinfo_window(start_text, end_text)
        with st.spinner('Waterinfo ophalen en controleren…'):
            base_rain = fetch_waterinfo(str(station_id), start, end, dt_seconds, timestamp_position)
        prepared = dict(key=rain_key, frame=base_rain)
        st.session_state.prepared_rain = prepared
    elif source == 'Eigen CSV' and csv_file is not None and (prepared is None or prepared['key'] != rain_key):
        base_rain = read_rain_csv(csv_file.getvalue(), int(csv_interval)*60, dt_seconds)
        prepared = dict(key=rain_key, frame=base_rain)
        st.session_state.prepared_rain = prepared
    if prepared is not None and prepared['key'] == rain_key:
        base_rain = prepared['frame']
        duration = pd.Timestamp(base_rain.attrs['window_end'])-pd.Timestamp(base_rain.attrs['window_start'])
        if duration > pd.Timedelta(hours=72):
            raise ValueError('De regenreeks is langer dan 72 uur. Kies zelf een kortere invoerperiode; er wordt niets afgekapt.')
        rain = scale_rain(base_rain, precipitation_factor)
except ValueError as exc:
    st.error(str(exc))
except Exception as exc:
    st.error('Neerslag kon niet worden geladen. Controleer de Waterinfo-verbinding of gebruik Eigen CSV.')
    with st.expander('Technische foutinformatie neerslag'):
        st.code(f'{type(exc).__name__}: {exc}')

if rain is not None:
    cols = st.columns(3)
    cols[0].metric('Oorspronkelijke buisom', f"{rain.attrs['original_total_mm']:.4f} mm")
    cols[1].metric('Geschaalde buisom', f"{rain.attrs['scaled_total_mm']:.4f} mm")
    cols[2].metric('Modeltijdstap', f'{timestep_minutes} min')
    st.write(f"Periode [begin, einde): **{local_time(rain.attrs['window_start'])} → "
             f"{local_time(rain.attrs['window_end'])}** (Belgische lokale tijd).")
    st.caption(f'{len(rain)} regenstappen; neerslagfactor {precipitation_factor:g}; runofffactor {runoff_factor:g}. '
               f'Daarna {tail_hours} uur droge uitloop (afgerond naar hele modelstappen).')
    st.vega_lite_chart(spec=chart_spec(
        rain.datum, rain.rf, 'Bui vóór berekening', 'Neerslag [mm/stap]',
        time_axis(rain.attrs['window_start'], rain.attrs['window_end']),
        interval_seconds=timestep_minutes*60), width='stretch', height=280)
else:
    st.info('Haal de neerslag op of upload een geldige CSV om de buisom te controleren en te berekenen.')

config = dict(code=code_identity(),
              inputs=dict(geodata=file_identity(geodata_file), catchments=file_identity(catchments_file),
                          inlets=file_identity(inlets_file),
                          measures=[file_identity(f) for f in (measures_files or [])]),
              rain=rain_settings, rain_identity=rain_identity(rain) if rain is not None else None,
              factors=dict(precipitation=precipitation_factor, runoff=runoff_factor,
                           slope=slope_factor, manning=manning_factor),
              model=dict(dt_s=dt_seconds, tail_seconds=float(tail_hours)*3600, A_threshold=A_threshold))
signature = fingerprint(config)

st.header('3. Berekenen')
ready = rain is not None and all(f is not None for f in [geodata_file, catchments_file, inlets_file])
if not all(f is not None for f in [geodata_file, catchments_file, inlets_file]):
    st.info('Upload alle drie de gebiedsbestanden bij Invoer.')
run_clicked = st.button('Berekenen', type='primary', disabled=not ready, key='calculate')
if run_clicked and ready:
    st.session_state.logs = []
    st.session_state.run_error = None
    progress = st.progress(0.0, text='Invoer inlezen…')
    status = st.empty()

    def log(message):
        st.session_state.logs.append(str(message))
        status.write(str(message))

    def update_progress(step, total, current_time):
        progress.progress((step+1)/total,
                          text=f'Tijdstap {step+1}/{total} · {local_time(current_time)}')

    try:
        log('Gebiedsbestanden inlezen…')
        geodata, catchments, inlets, measures = read_inputs(geodata_file, catchments_file, inlets_file, measures_files)
        log('GIS-invoer controleren en maatregelen verwerken…')
        inputs, crs, L = preprocess_geodata(geodata, catchments, inlets, measures,
                                            slope_factor=slope_factor, manning_factor=manning_factor,
                                            runoff_factor=runoff_factor, apply_measures=apply_maatregelen, log=log)
        reservoirs = build_reservoirs(catchments, inputs, L, crs)
        result = run_model(catchments, inputs, L, reservoirs, rain, A_threshold,
                           update_progress=update_progress, tail_seconds=float(tail_hours)*3600, log=log)
        log(f"Waterbalans: restfout {result.balance['residual_m3']:.3g} m³.")
        log('Reeksen, diepterasters en runmetadata naar ZIP schrijven…')
        completed = prepare_results(result, rain, reservoirs, inputs, catchments, config, signature, st.session_state.logs)
        # Commit the entire snapshot only after calculation AND export succeed.
        st.session_state.completed_run = completed
        st.session_state.run_history = dict(st.session_state.run_history, **{scenario(config): completed})
        progress.progress(1.0, text='Berekening en ZIP gereed')
        status.empty()
    except Exception as exc:
        message = str(exc) if isinstance(exc, (ValueError, ArithmeticError)) else (
            f'Berekening niet voltooid. Controleer de invoerbestanden. {type(exc).__name__}: {exc}')
        st.session_state.run_error = message
        log(f'Run geblokkeerd: {message}')
        progress.empty()

if st.session_state.run_error:
    st.error(st.session_state.run_error)
with st.expander('Logboek laatste berekenpoging'):
    st.text('\n'.join(st.session_state.logs) or 'Nog geen berekening uitgevoerd.')

st.header('4. Resultaten')
saved = st.session_state.completed_run
if saved is None:
    st.info('Na een geslaagde berekening verschijnen hier de grafieken en de resultaat-ZIP.')
else:
    stale = saved['signature'] != signature
    if stale:
        st.warning('Resultaten verouderd: de invoerinstellingen zijn gewijzigd. Grafieken en ZIP horen bij '
                   'de vorige geslaagde run. Klik op Berekenen om de nieuwe invoer door te rekenen.')
    else:
        st.success('Resultaten horen bij de huidige invoerinstellingen.')
    run = saved['result']
    saved_rain = saved['rain']
    st.caption(f"Run {saved['signature'][:12]} · {local_time(saved['metadata']['created_at_utc'])} · "
               f"{saved_rain.attrs['scaled_total_mm']:.4f} mm · modeltijdstap {run.dt_s/60:g} min.")
    st.download_button('Download ZIP vorige run (verouderd)' if stale else 'Download ZIP met resultaten',
                       data=saved['zip_bytes'], file_name=saved['filename'], mime='application/zip',
                       on_click='ignore', key='results_zip')
    names = saved['names']
    if st.session_state.get('result_area') not in names:
        st.session_state.result_area = sorted(names)[0]
    cid = st.selectbox('Stroomgebied', sorted(names), format_func=lambda c: names[c], key='result_area')
    name = names[cid]
    pair = matching_runs(st.session_state.run_history, config)
    if pair is not None:
        st.subheader('Runvergelijking')
        st.caption('Laatste geslaagde run zonder en met maatregelen, met dezelfde overige invoer. '
                   'Verschil [%] = (met maatregelen − zonder maatregelen) / zonder maatregelen × 100. '
                   'Bij een basiswaarde van 0: n.v.t.')
        st.table(style_comparison_table(comparison_table(pair, cid)))
        st.caption('Tijd tot piekdebiet: uren vanaf de simulatiestart tot het intervalbegin van de eerste '
                   'hoogste debietwaarde. Bij uitsluitend nuldebiet: n.v.t.')
    elif all(label in st.session_state.run_history for label in SCENARIOS):
        st.info('Geen runvergelijking: gebiedsbestanden, periode, neerslag of overige modelinstellingen '
                'verschillen tussen de bewaarde runs of met de huidige invoer. Bereken beide scenario’s '
                'met dezelfde invoer; alleen de maatregelen mogen verschillen.')
    else:
        st.info('Voor runvergelijking: bereken eenmaal zonder en eenmaal met maatregelen, '
                'met dezelfde gebiedsbestanden, periode, neerslag en overige modelinstellingen. '
                'De laatste geslaagde run van elk scenario blijft tijdens deze sessie bewaard.')
    st.vega_lite_chart(spec=result_charts(run, saved_rain, saved['depths'], cid, name,
                                         comparison=pair), width='stretch')
    st.caption('Alle grafieken hebben dezelfde tijdas in Belgische lokale tijd (zomer-/wintertijd). '
               'Neerslagbalken beslaan het modelinterval; debiet staat op intervalbegin '
               'en waterdiepte geldt aan intervaleinde. '
               'Waterdiepte is de maximale diepte binnen het geselecteerde gebied.')
    st.caption('ZIP: debieten, stapvolumes, peilen, regen, maximale diepte-GeoTIFFs per gebied en runmetadata. '
               'De CSV-tijdstempels zijn intervalbegin in UTC; peilen gelden aan intervaleinde.')
    st.subheader('Kaart maximale waterdiepte')
    if cid not in saved.get('map_areas', {}):
        st.info('Bereken opnieuw om de kaart toe te voegen aan deze eerder opgeslagen run.')
    else:
        try:
            st.iframe(result_map_html(saved['zip_bytes'], cid, saved['map_areas'][cid]), height=620)
        except Exception as exc:
            st.warning('De kaart kon niet worden opgebouwd. De resultaten blijven beschikbaar in de ZIP.')
            with st.expander('Technische foutinformatie kaart'):
                st.code(f'{type(exc).__name__}: {exc}')
        st.caption('Maximale waterdiepte over de volledige simulatie, inclusief de begintoestand. '
                   'Geel: stroomgebied; rood: uitlaatpunt. Schakel GRB in via de laagkeuze rechtsboven. '
                   'Orthofoto: meest recente beschikbare winteropname per locatie (Digitaal Vlaanderen). '
                   'De achtergrondkaarten vereisen internet; de kaartweergave kan zijn verkleind, '
                   'de GeoTIFF in de ZIP behoudt de oorspronkelijke resolutie.')
    with st.expander('Waterbalans en runinstellingen'):
        st.json(saved['metadata'])
    st.caption('Resultaten blijven bewaard tijdens deze sessie. Download de ZIP vóór het sluiten van de sessie.')
