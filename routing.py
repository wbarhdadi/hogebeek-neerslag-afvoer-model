"""Validated Whitebox D8 topology and time-varying, volume-conserving routing."""
from collections import deque
from dataclasses import dataclass
import numpy as np

DIRECTION_MAP = {1: (-1, 1), 2: (0, 1), 4: (1, 1), 8: (1, 0),
                 16: (1, -1), 32: (0, -1), 64: (-1, -1), 128: (-1, 0)}


@dataclass
class D8Topology:
    shape: tuple
    valid: np.ndarray
    downstream: np.ndarray
    order: np.ndarray
    outlet: int
    distance_factor: np.ndarray

    @classmethod
    def build(cls, flow_dir, outlet, mask=None):
        fd = np.asarray(flow_dir, float)
        mask = np.isfinite(fd) if mask is None else np.asarray(mask, bool)
        if mask.shape != fd.shape or fd.ndim != 2:
            raise ValueError('D8-raster en gebiedsmasker moeten hetzelfde tweedimensionale grid hebben.')
        height, width = fd.shape
        r, c = outlet
        if not (0 <= r < height and 0 <= c < width) or not mask[r, c]:
            raise ValueError(f'Uitlaat {outlet} valt niet op een geldige gebiedscel.')
        outlet_flat = r*width+c
        if fd[r, c] != 0 and fd[r, c] not in DIRECTION_MAP:
            raise ValueError(f'Ongeldige D8-code {fd[r, c]} op de uitlaat {outlet}.')
        valid = np.flatnonzero(mask)
        downstream = np.full(fd.size, -1, dtype=np.int64)
        distances = np.ones(fd.size)
        flat = fd.ravel()
        for i in valid:
            if i == outlet_flat:
                continue  # The outlet is the absorbing end of this local catchment.
            if flat[i] not in DIRECTION_MAP:
                raise ValueError(f'Ongeldige D8-code {flat[i]} op cel {divmod(int(i), width)}; alleen de uitlaat mag stoppen.')
            dr, dc = DIRECTION_MAP[flat[i]]
            nr, nc = i//width+dr, i%width+dc
            if not (0 <= nr < height and 0 <= nc < width) or not mask[nr, nc]:
                raise ValueError(f'D8-route verlaat het gebied bij cel {divmod(int(i), width)}, vóór uitlaat {outlet}.')
            downstream[i] = nr*width+nc
            distances[i] = np.hypot(dr, dc)
        degree = np.bincount(downstream[downstream >= 0], minlength=fd.size)
        queue = deque(int(i) for i in valid if degree[i] == 0)
        order = []
        while queue:
            i = queue.popleft()
            order.append(i)
            j = downstream[i]
            if j >= 0:
                degree[j] -= 1
                if degree[j] == 0:
                    queue.append(int(j))
        if len(order) != len(valid):
            bad = valid[degree[valid] > 0][0]
            raise ValueError(f'D8-cyclus vanaf cel {divmod(int(bad), width)}; run geblokkeerd.')
        return cls(fd.shape, valid, downstream, np.asarray(order, dtype=np.int64), outlet_flat, distances)

    def accumulate_volume(self, local):
        result = np.nan_to_num(np.asarray(local, float), nan=0.).ravel().copy()
        _upstream_sum(result, self.downstream, self.order)
        return result.reshape(self.shape)

    def accumulate_time(self, local_time):
        times = np.asarray(local_time, float).ravel()
        result = np.full(times.shape, np.nan)
        result[self.outlet] = 0.
        _downstream_sum(result, times, self.downstream, self.order)
        return result.reshape(self.shape)


def _upstream_sum(result, downstream, order):
    for i in order:
        j = downstream[i]
        if j >= 0:
            result[j] += result[i]


def _downstream_sum(result, times, downstream, order):
    for k in range(len(order)-1, -1, -1):
        i = order[k]
        j = downstream[i]
        if j >= 0:
            result[i] = times[i]+result[j]


def accumulate_travel_time(travel_time, flow_dir, outlet):
    topology = D8Topology.build(flow_dir, outlet)
    return topology.accumulate_time(travel_time)


def route_Q_channel(fd_catchment, dtm_catchment, channel_mask, inlet_flow_dict,
                    inlet_indices_dict, effective_rain_depth, outlet_indices, L):
    """Cumulative m3/step including all upstream hillslopes and the cell itself.

    Topological order handles tributary joins exactly once. Terrain elevation is
    deliberately not used to order a breached D8 graph.
    """
    topology = D8Topology.build(fd_catchment, outlet_indices)
    volume = np.nan_to_num(effective_rain_depth, nan=0.)*.001*L**2
    for source, rc in inlet_indices_dict.items():
        volume[rc] += inlet_flow_dict.get(source, 0.)
    accumulated = topology.accumulate_volume(volume)
    return np.where(channel_mask, accumulated, 0.)


def tv_convolve_next(PU, timestep):
    """PU[lag, source_step] is m3 arriving at source_step + lag (zero-based)."""
    PU = np.asarray(PU, float)
    if PU.ndim != 2 or timestep < 0:
        raise ValueError('Responsmatrix moet tweedimensionaal zijn; tijdstap moet >= 0 zijn.')
    lag = np.arange(PU.shape[0])
    column = timestep-lag
    valid = (column >= 0) & (column < PU.shape[1])
    return float(PU[lag[valid], column[valid]].sum())


class ArrivalQueue:
    """Sparse future m3 totals keyed by absolute model step; no lag truncation."""
    def __init__(self):
        self.pending = {}

    def add(self, timestep, delays_s, volumes, dt_s):
        delays, volumes = np.broadcast_arrays(np.asarray(delays_s, float), np.asarray(volumes, float))
        if not np.isfinite(volumes).all() or (volumes < 0).any() or not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError('Ongeldige routevolumes of tijdstap.')
        active = volumes > 0
        if not np.isfinite(delays[active]).all() or (delays[active] < 0).any():
            raise ValueError('Water heeft geen eindige reistijd naar de uitlaat (controleer nulhelling/ruwheid/D8).')
        # Same isochrone convention as the original method: [lag*dt,(lag+1)*dt).
        indices = np.floor(delays[active]/dt_s)
        if not np.isfinite(indices).all():
            raise ValueError('Reistijd/tijdstap-verhouding is niet eindig.')
        # Python integer keys preserve even the enormous (but finite) lags of
        # tiny residual Q outflows. int64 would overflow after a long dry tail.
        lags, inverse = np.unique(indices, return_inverse=True)
        sums = np.bincount(inverse, weights=volumes[active])
        for lag, volume in zip(lags, sums):
            arrival = timestep+int(lag)
            self.pending[arrival] = self.pending.get(arrival, 0.)+float(volume)

    def pop(self, timestep):
        return self.pending.pop(timestep, 0.)

    @property
    def volume(self):
        return float(sum(self.pending.values()))


def travel_times(inp, local_depth_mm, inlet_volumes, dt_s, L, A_threshold):
    """Melesse & Graham (2004), equations 1/4, SI units, pp. 865-866.

    Overland intensity is effective depth / dt in m/s. Flowing cells with zero
    local runoff use upstream unit discharge/width for transport, so runoff-zero
    measures cannot destroy upstream water. Zero slope is a visible block when
    a positive volume needs that route, rather than an arbitrary slope floor.
    """
    topology = inp['topology']
    volume = np.nan_to_num(local_depth_mm, nan=0.)*.001*L**2
    for source, rc in inp['inlet_dict'].items():
        volume[rc] += inlet_volumes.get(source, 0.)
    cumulative_q = topology.accumulate_volume(volume)/dt_s
    slope, n = inp['S_catchment'], inp['manning_catchment']
    distance = L*topology.distance_factor.reshape(topology.shape)
    intensity = np.nan_to_num(local_depth_mm, nan=0.)*.001/dt_s
    channel = inp['fa_catchment'] >= A_threshold
    for source, rc in inp['inlet_dict'].items():
        if not channel[rc]:
            raise ValueError(f'Inlaat uit gebied {source} ligt buiten kanaal; verlaag A_threshold of controleer invoer.')
    times = np.full(topology.shape, np.inf)
    wet = (intensity > 0) & (slope > 0)
    times[wet] = distance[wet]**.6*n[wet]**.6/(intensity[wet]**.4*slope[wet]**.3)
    transport = (channel | (intensity == 0)) & (cumulative_q > 0) & (slope > 0)
    width = np.where(channel, L/2., L)
    velocity = ((np.sqrt(slope[transport])/n[transport])*(cumulative_q[transport]/width[transport])**(2/3))**(3/5)
    times[transport] = distance[transport]/velocity
    return topology.accumulate_time(times), volume
