"""Single calculation path used by the Streamlit app and regression tests."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from routing import ArrivalQueue, travel_times

REL_TOL = 1e-6
ABS_TOL_M3 = 1e-8


@dataclass
class ModelResult:
    volumes: dict
    waterlevels: dict
    times: pd.DatetimeIndex
    dt_s: float
    balance: dict
    balance_per_catchment: dict
    metadata: dict

    @property
    def discharges(self):
        return {cid: volume/self.dt_s for cid, volume in self.volumes.items()}

    def __iter__(self):
        # Compatibility with the original app's volume-returning function.
        return iter((self.volumes, self.waterlevels, self.times, self.dt_s))


def compute_effective_recharge(P_current, ro_catchment):
    return np.asarray(ro_catchment, float)/100.*float(P_current)


def _network(inputs):
    destination = {}
    for cid, inp in inputs.items():
        for source in inp['inlet_dict']:
            if source not in inputs or source == cid:
                raise ValueError('Onbekende bron of zelfverbinding in netwerk.')
            if source in destination:
                raise ValueError('Meerdere bestemmingen voor één bronvolume zijn niet ondersteund.')
            destination[source] = cid
    for cid in inputs:
        seen = set()
        while cid in destination:
            if cid in seen:
                raise ValueError('Netwerk bevat een cyclus.')
            seen.add(cid)
            cid = destination[cid]
    return destination


def run_model(catchments, inputs, L, reservoirs, rf_timeseries, A_threshold,
              update_progress=None, tail_seconds=72*3600, log=print):
    """Run interval-start rainfall including step zero and explicit dry tail.

    A reservoir's integrated outflow enters its downstream inlet in the next
    step (the original explicit coupling), then follows that catchment's D8
    travel time. The last-step inter-catchment volume is accounted as in transit.
    H storage below its crest is intentionally retained after rain stops.
    """
    if set(inputs) != set(reservoirs):
        raise ValueError('Gebieden en reservoirs komen niet overeen.')
    if not np.isfinite(L) or L <= 0 or not np.isfinite(A_threshold) or A_threshold < 0:
        raise ValueError('Ongeldige celgrootte of kanaaldrempel.')
    dates = pd.DatetimeIndex(rf_timeseries['datum'])
    if dates.tz is None or dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError('Modelregen vereist unieke, oplopende tijdstippen met tijdzone.')
    P = rf_timeseries['rf'].to_numpy(float)
    if not len(P) or not np.isfinite(P).all() or (P < 0).any():
        raise ValueError('Modelregen is leeg, negatief of niet eindig.')
    dt_s = rf_timeseries.attrs.get('dt_s')
    if dt_s is None:
        if len(dates) < 2:
            raise ValueError('Bij één regenstap moet attrs[dt_s] expliciet zijn ingesteld.')
        dt_s = (dates[1]-dates[0]).total_seconds()
    dt_s = float(dt_s)
    if dt_s <= 0 or not np.isfinite(dt_s):
        raise ValueError('Modeltijdstap moet positief en eindig zijn.')
    if len(dates) > 1 and not np.allclose(np.diff(dates.asi8)/1e9, dt_s, rtol=0, atol=1e-6):
        raise ValueError('Modelregen heeft gaten of een afwijkende intervalduur.')
    if not np.isfinite(tail_seconds) or tail_seconds < 0:
        raise ValueError('Uitloopduur moet eindig en niet-negatief zijn.')
    tail_steps = int(np.ceil(tail_seconds/dt_s))
    n_steps = len(P)+tail_steps
    times = pd.date_range(dates[0], periods=n_steps, freq=pd.Timedelta(seconds=dt_s))
    destination = _network(inputs)
    queues = {cid: ArrivalQueue() for cid in inputs}
    volumes = {cid: np.zeros(n_steps) for cid in inputs}
    levels = {cid: np.zeros(n_steps) for cid in inputs}
    initial = {cid: float(reservoirs[cid].S) for cid in inputs}
    rain_volume = dict.fromkeys(inputs, 0.)
    received = dict.fromkeys(inputs, 0.)
    initial_levels = {cid: reservoirs[cid].get_waterlevel_from_volume(initial[cid]) for cid in inputs}
    # Check inlet/channel alignment even for a completely dry run.
    for cid, inp in inputs.items():
        for source, rc in inp['inlet_dict'].items():
            if inp['fa_catchment'][rc] < A_threshold:
                raise ValueError(f'Gebied {cid}, inlaat {source}: A_threshold te hoog; geen kanaalcel.')
    log(f'Model: {len(P)} regenstappen + {tail_steps} droge uitloopstappen; Δt={dt_s:g} s.')
    for step in range(n_steps):
        rainfall = P[step] if step < len(P) else 0.
        if update_progress is not None:
            update_progress(step, n_steps, times[step])
        for cid, inp in inputs.items():
            incoming = {source: volumes[source][step-1] if step else 0. for source in inp['inlet_dict']}
            received[cid] += sum(incoming.values())
            effective = compute_effective_recharge(rainfall, inp['ro_catchment'])
            rain_volume[cid] += float(np.nansum(effective)*.001*L**2)
            if rainfall > 0 or any(value > 0 for value in incoming.values()):
                delay, local_volume = travel_times(inp, effective, incoming, dt_s, L, A_threshold)
                try:
                    queues[cid].add(step, delay, local_volume, dt_s)
                except ValueError as exc:
                    raise ValueError(f'Gebied {cid}, stap {step}: {exc}') from exc
            # Always service old arrivals and storage, including during drought.
            arrival = queues[cid].pop(step)
            volumes[cid][step] = reservoirs[cid].transfer(arrival, dt_s)
            levels[cid][step] = reservoirs[cid].get_waterlevel_from_volume(reservoirs[cid].S)
    per_area = {}
    for cid in inputs:
        per_area[cid] = dict(initial_storage_m3=initial[cid], runoff_m3=rain_volume[cid],
                             received_internal_m3=received[cid], outflow_m3=float(volumes[cid].sum()),
                             final_storage_m3=float(reservoirs[cid].S), routing_in_transit_m3=queues[cid].volume)
        b = per_area[cid]
        b['residual_m3'] = b['initial_storage_m3']+b['runoff_m3']+b['received_internal_m3']-b['outflow_m3']-b['final_storage_m3']-b['routing_in_transit_m3']
        scale = initial[cid]+rain_volume[cid]+received[cid]
        if abs(b['residual_m3']) > ABS_TOL_M3+REL_TOL*scale:
            raise ArithmeticError(f'Waterbalans gebied {cid} sluit niet: {b["residual_m3"]} m³.')
    balance = dict(initial_storage_m3=sum(initial.values()), effective_rain_m3=sum(rain_volume.values()),
                   external_outflow_m3=sum(float(v.sum()) for cid, v in volumes.items() if cid not in destination),
                   internal_transfer_m3=sum(float(volumes[cid].sum()) for cid in destination),
                   final_storage_m3=sum(float(r.S) for r in reservoirs.values()),
                   routing_in_transit_m3=sum(q.volume for q in queues.values()),
                   network_in_transit_m3=sum(float(volumes[cid][-1]) for cid in destination))
    balance['residual_m3'] = (balance['initial_storage_m3']+balance['effective_rain_m3']-balance['external_outflow_m3']
                              -balance['final_storage_m3']-balance['routing_in_transit_m3']-balance['network_in_transit_m3'])
    scale = balance['initial_storage_m3']+balance['effective_rain_m3']
    balance['relative_error'] = abs(balance['residual_m3'])/scale if scale else 0.
    if abs(balance['residual_m3']) > ABS_TOL_M3+REL_TOL*scale:
        raise ArithmeticError(f'Netwerkwaterbalans sluit niet: {balance["residual_m3"]} m³.')
    metadata = dict(dt_s=dt_s, rain_steps=len(P), dry_tail_steps=tail_steps,
                    rain_mm=float(P.sum()), rain_metadata=rf_timeseries.attrs,
                    network_transfer='one model step, followed by receiving catchment travel time',
                    response_lags='floor(travel_time_s / dt_s); no cap',
                    level_timestamps='end of interval; row timestamp is interval start',
                    initial_levels_m=initial_levels,
                    units=dict(volumes='m3/step', discharges='m3/s', waterlevel='m in DTM vertical reference (unverified)', waterdepth='m above DTM_5m'),
                    terrain=dict(routing='D8flowdirection / D8slope / breachedburnedDTM_5m', storage='DTM_5m'),
                    geen_policy='H reservoir, project owner confirmed 2026-10-28',
                    relative_tolerance=REL_TOL, absolute_tolerance_m3=ABS_TOL_M3)
    return ModelResult(volumes, levels, times, dt_s, balance, per_area, metadata)
