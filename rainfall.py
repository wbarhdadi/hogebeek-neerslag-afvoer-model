"""Rain contract: timezone-aware interval starts, millimetres per model step.

Resampling integrates interval overlap; it never replaces missing rain with zero.
Waterinfo P.1/P.5 spacing comes from the series definition. Timestamp position is
an explicit model convention (start by default), not inferred from numeric values.
"""
from dataclasses import dataclass, asdict
import csv
import hashlib
from io import StringIO
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RainMetadata:
    unit: str
    interval_seconds: float
    timestamp_position: str  # start or end
    source: str


def aware_timestamp(value):
    ts = pd.Timestamp(value)
    if pd.isna(ts) or ts.tzinfo is None:
        raise ValueError(f'Tijdstip {value!r}: een expliciete tijdzone/UTC-offset is verplicht.')
    return ts.tz_convert('UTC')


def normalize_rain(timestamps, values, metadata, start, end, dt_seconds):
    """Integrate piecewise-constant intensities / uniformly distributed amounts.

    Source intervals may straddle model boundaries. A complete, ordered source
    series is required, including dry intervals. The window is [start, end).
    """
    start, end = aware_timestamp(start), aware_timestamp(end)
    dt = float(dt_seconds)
    width = float(metadata.interval_seconds)
    if not np.isfinite([dt, width]).all() or dt <= 0 or width <= 0 or end <= start:
        raise ValueError('Intervalduur en modelstap moeten positief zijn; einde moet na begin liggen.')
    if metadata.unit not in ('mm', 'mm/h') or metadata.timestamp_position not in ('start', 'end'):
        raise ValueError('Onbekende regenmetadata: verwacht mm of mm/h en timestamp_position start/end.')
    n_float = (end-start).total_seconds()/dt
    times = []
    for row, value in enumerate(timestamps, 2):
        try:
            times.append(aware_timestamp(value))
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Regen regel {row}: {exc}') from exc
    values = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=float)
    if len(times) != len(values) or not len(times):
        raise ValueError('Lege of onvolledige regenreeks.')
    invalid = ~np.isfinite(values) | (values < 0)
    if invalid.any():
        raise ValueError(f'Ongeldige/ontbrekende/negatieve regen op regels {(np.flatnonzero(invalid)+2).tolist()[:20]}.')
    stamp = pd.DatetimeIndex(times)
    duplicates = stamp.duplicated(keep=False)
    if duplicates.any():
        raise ValueError(f'Dubbele regentijdstippen op regels {(np.flatnonzero(duplicates)+2).tolist()[:20]}.')
    seconds = np.array([(t-start).total_seconds() for t in stamp])
    spacing = np.diff(seconds)
    if (spacing <= 0).any():
        rows = (np.flatnonzero(spacing <= 0)+2).tolist()[:20]
        raise ValueError(f'Regen moet chronologisch geordend zijn; controleer regels {rows} en de volgende regel.')
    gaps = ~np.isclose(spacing, width, rtol=0, atol=1e-6)
    if gaps.any():
        raise ValueError(f'Gaten of afwijkende regenintervallen tussen regels {(np.flatnonzero(gaps)+2).tolist()[:20]} en de volgende regel.')
    if not np.isclose(n_float, round(n_float), rtol=0, atol=1e-9):
        raise ValueError('De gevraagde periode bevat een onvolledige modelstap; pas periode of stapduur aan.')
    if metadata.timestamp_position == 'end':
        seconds -= width
    duration = (end-start).total_seconds()
    if seconds[0] > 1e-6 or seconds[-1] + width < duration-1e-6:
        raise ValueError('Regenreeks dekt de gevraagde periode niet volledig; geen stilzwijgende afkapping.')
    amounts = values * width/3600 if metadata.unit == 'mm/h' else values
    result = np.zeros(round(n_float))
    for left, amount in zip(seconds, amounts):
        right = left+width
        a, b = max(0., left), min(duration, right)
        if b <= a:
            continue
        first, last = int(a//dt), min(len(result)-1, int(np.ceil(b/dt))-1)
        for k in range(first, last+1):
            result[k] += amount * max(0., min(b, (k+1)*dt)-max(a, k*dt))/width
    frame = pd.DataFrame({'datum': pd.date_range(start, periods=len(result), freq=pd.Timedelta(seconds=dt)), 'rf': result})
    frame.attrs.update(dt_s=dt, rain_metadata=asdict(metadata), window_start=start.isoformat(), window_end=end.isoformat(), interval='[start, end)')
    return frame


def waterinfo_metadata(record, timestamp_position='start'):
    unit = str(record.get('ts_unitsymbol', '')).strip()
    series = str(record.get('ts_name', '')).strip()
    # Explicit known source series; reject an unrecognized source rather than
    # estimating its cadence from a series that may contain missing rows.
    intervals = {'P.1': 60., 'P.5': 300.}
    if series not in intervals or unit not in ('mm/h', 'mm'):
        raise ValueError(f'Niet ondersteunde Waterinfo-bronmetadata: {series!r}, {unit!r}.')
    return RainMetadata(unit, intervals[series], timestamp_position, f"Waterinfo ts_id={record.get('ts_id')} {series}")


def fetch_waterinfo(ts_id, start, end, dt_seconds, timestamp_position='start'):
    from pywaterinfo import Waterinfo
    client = Waterinfo('vmm')
    meta = client.get_timeseries_list(ts_id=str(ts_id), returnfields='ts_id,ts_name,ts_unitsymbol,parametertype_name,station_name')
    if len(meta) != 1:
        raise ValueError('Waterinfo leverde geen unieke bronmetadata.')
    metadata = waterinfo_metadata(meta.iloc[0], timestamp_position)
    start, end = aware_timestamp(start), aware_timestamp(end)
    # Fetch both boundaries, then normalize exactly the requested half-open window.
    raw = client.get_timeseries_values(ts_id=str(ts_id), start=start.to_pydatetime(), end=end.to_pydatetime())
    if not {'Timestamp', 'Value'}.issubset(raw.columns):
        raise ValueError('Waterinfo mist Timestamp en/of Value.')
    frame = normalize_rain(raw.Timestamp, raw.Value, metadata, start, end, dt_seconds)
    frame.attrs.update(source_identity=dict(
        ts_id=str(ts_id), station_name=str(meta.iloc[0].get('station_name', '')),
        raw_sha256=hashlib.sha256(raw.to_csv(index=False).encode('utf-8')).hexdigest(),
        retrieved_at_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    return frame


def read_rain_csv(data, interval_seconds, dt_seconds):
    """Read the entire UTF-8 CSV; physical line numbers match validation errors.

    One record per line, exact columns, interval amounts and explicit offsets.
    Blank/multiline records are rejected rather than changing line numbering.
    """
    try:
        text = data.decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        raise ValueError('CSV moet UTF-8 zijn; sla het bestand opnieuw op als CSV UTF-8.') from exc
    reader = csv.reader(StringIO(text, newline=''), strict=True)
    times, values = [], []
    try:
        if next(reader, None) != ['timestamp', 'precipitation_mm']:
            raise ValueError('CSV regel 1: verwacht exact timestamp,precipitation_mm (komma als scheidingsteken).')
        for row_number, row in enumerate(reader, 2):
            if reader.line_num != row_number or len(row) != 2:
                raise ValueError(f'CSV regel {row_number}: verwacht twee velden op één regel; lege regels zijn niet toegestaan.')
            try:
                times.append(aware_timestamp(row[0]))
            except (ValueError, TypeError) as exc:
                raise ValueError(f'CSV regel {row_number}: {exc}') from exc
            values.append(row[1])
    except csv.Error as exc:
        raise ValueError(f'CSV regel {reader.line_num}: ongeldige CSV-opmaak ({exc}).') from exc
    if not times:
        raise ValueError('CSV bevat geen regen; voeg gegevens toe vanaf regel 2.')
    width = float(interval_seconds)
    if not np.isfinite(width) or width <= 0:
        raise ValueError('CSV-intervalduur moet positief en eindig zijn.')
    # Validate ordering and values before deriving the full window, so a reversed
    # series also reports its offending source lines.
    start = min(times)
    end = max(times)+pd.Timedelta(seconds=width)
    frame = normalize_rain(times, values, RainMetadata('mm', width, 'start', 'Eigen CSV'),
                           start, end, dt_seconds)
    frame.attrs['source_identity'] = dict(csv_sha256=hashlib.sha256(data).hexdigest())
    return frame


def scale_rain(frame, factor):
    """Apply the independent precipitation factor once, after normalization."""
    factor = float(factor)
    if not np.isfinite(factor) or factor < 0:
        raise ValueError('Neerslagfactor moet eindig en niet-negatief zijn.')
    if 'precipitation_factor' in frame.attrs:
        raise ValueError('Neerslagfactor is al toegepast; schaal altijd de oorspronkelijke genormaliseerde regen.')
    scaled = frame.copy(deep=True)
    scaled['rf'] = frame.rf*factor
    if not np.isfinite(scaled.rf).all() or not np.isfinite(scaled.rf.sum()):
        raise ValueError('Geschaalde neerslag is te groot; kies een kleinere neerslagfactor.')
    scaled.attrs.update(precipitation_factor=factor, original_total_mm=float(frame.rf.sum()),
                        scaled_total_mm=float(scaled.rf.sum()))
    return scaled
