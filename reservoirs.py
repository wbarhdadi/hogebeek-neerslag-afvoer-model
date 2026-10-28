"""Reservoir storage and transfer, all volumes m3, time seconds, levels metres."""
from pathlib import Path
import math
import numpy as np
import rasterio


class QReservoir:
    def __init__(self, catchment_id, dtm_array, pixel_area, transform, crs, A, Qmax):
        self.catchment_id = int(catchment_id)
        self.dtm = np.asarray(dtm_array, float).copy()
        self.pixel_area, self.transform, self.crs = float(pixel_area), transform, crs
        self.A, self.Qmax = float(A), float(Qmax)
        if not np.isfinite([self.A, self.Qmax, self.pixel_area]).all() or self.A <= 0 or self.Qmax < 0 or self.pixel_area <= 0:
            raise ValueError(f'Gebied {catchment_id}: ongeldige K/Q-capaciteit/pixeloppervlakte.')
        self.t, self.S = 0., 0.
        self._setup_waterlevel_volume_relationship()
        self.H = self.get_waterlevel_from_volume(0)

    def _setup_waterlevel_volume_relationship(self):
        z = np.sort(self.dtm[np.isfinite(self.dtm)])
        if not len(z):
            raise ValueError(f'Gebied {self.catchment_id}: geen geldig fysiek terrein.')
        self._z = z
        self._zsum = np.r_[0., np.cumsum(z)]
        # Exact piecewise-linear hypsometry at every distinct elevation. No
        # 100-level approximation or clamping above the highest terrain cell.
        self._levels, count = np.unique(z, return_counts=True)
        below = np.r_[0, np.cumsum(count)[:-1]]
        self._volumes = self.pixel_area * (below*self._levels-self._zsum[below])
        self._wet_counts = np.cumsum(count)

    def get_volume_from_waterlevel(self, H):
        H = float(H)
        if not np.isfinite(H):
            raise ValueError('Peil moet eindig zijn.')
        n = np.searchsorted(self._z, H)
        return max(0., float(self.pixel_area*(n*H-self._zsum[n])))

    def get_waterlevel_from_volume(self, S):
        S = float(S)
        if not np.isfinite(S) or S < 0:
            raise ValueError('Opslag moet eindig en niet-negatief zijn.')
        i = max(0, np.searchsorted(self._volumes, S, side='right')-1)
        return float(self._levels[i] + (S-self._volumes[i])/(self._wet_counts[i]*self.pixel_area))

    def _validate_transfer(self, Vin, dt):
        if not np.isfinite([Vin, dt, self.S]).all() or Vin < 0 or dt <= 0 or self.S < 0:
            raise ValueError('Instroom/opslag moet niet-negatief en eindig zijn; stapduur moet positief zijn.')

    def transfer(self, Vin, tres):
        """Exact integration of dS/dt = I - min(A*S, Qmax), constant I=Vin/dt.

        Solve either linear or capacity-limited segment, including the crossing
        of Scrit=Qmax/A inside the step. Return integrated outflow, not end Q.
        """
        Vin, dt = float(Vin), float(tres)
        self._validate_transfer(Vin, dt)
        initial, rate, cap, a = self.S, Vin/dt, self.Qmax, self.A
        critical = cap/a
        def linear(s, time):
            return s*math.exp(-a*time) + rate*(-math.expm1(-a*time))/a
        if cap == 0:
            final = initial+Vin
        elif initial >= critical:
            if rate >= cap:
                final = initial+(rate-cap)*dt
            else:
                crossing = (initial-critical)/(cap-rate)
                final = initial+(rate-cap)*dt if crossing >= dt else linear(critical, dt-crossing)
        else:
            if rate <= cap:
                final = linear(initial, dt)
            else:
                crossing = math.log((rate-a*initial)/(rate-cap))/a
                final = linear(initial, dt) if crossing >= dt else critical+(rate-cap)*(dt-crossing)
        available = initial+Vin
        outflow = min(available, cap*dt, max(0., available-final))
        self.S = max(0., available-outflow)
        self.H = self.get_waterlevel_from_volume(self.S)
        self.t += dt
        return outflow

    def export_waterdepth(self, out_fp, waterlevel=None):
        level = self.get_waterlevel_from_volume(self.S) if waterlevel is None else float(waterlevel)
        arr = np.where(np.isfinite(self.dtm), np.maximum(0, level-self.dtm), -9999.).astype('float32')
        out_fp = Path(out_fp)
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_fp, 'w', driver='GTiff', height=arr.shape[0], width=arr.shape[1], count=1,
                           dtype='float32', crs=self.crs, transform=self.transform, nodata=-9999.) as dest:
            dest.write(arr, 1)
            dest.update_tags(units='m above physical terrain', terrain='DTM_5m')
        return out_fp


class HReservoir(QReservoir):
    def __init__(self, catchment_id, dtm_array, pixel_area, transform, crs, Hmax):
        super().__init__(catchment_id, dtm_array, pixel_area, transform, crs, 1., 0.)
        self.Hmax = float(Hmax)
        self.Smax = self.get_volume_from_waterlevel(Hmax)

    def transfer(self, Vin, tres):
        Vin, dt = float(Vin), float(tres)
        self._validate_transfer(Vin, dt)
        available = self.S+Vin
        outflow = max(0., available-self.Smax)
        self.S = available-outflow
        self.H = self.get_waterlevel_from_volume(self.S)
        self.t += dt
        return outflow


def build_reservoirs(catchments, inputs, L, crs):
    reservoirs = {}
    for row in catchments.itertuples():
        cid = int(row.uitstroompunt_nummer)
        args = (cid, inputs[cid]['dtm_catchment'], L**2, inputs[cid]['transform'], crs)
        kind = str(row.uitstroompunt_type).strip()
        if kind == 'Q':
            k = float(row.K_s)
            if not np.isfinite(k) or k <= 0:
                raise ValueError(f'Gebied {cid}: K_s moet positief en eindig zijn.')
            reservoirs[cid] = QReservoir(*args, A=1/k, Qmax=row.Q_cap_ms)
        elif kind in ('H', 'Geen'):
            # Project owner explicitly confirmed on 2026-10-28: Geen means H.
            reservoirs[cid] = HReservoir(*args, Hmax=row.outlet_mtaw)
        else:
            raise ValueError(f'Gebied {cid}: onopgelost reservoirtype {kind!r}; run geblokkeerd.')
    return reservoirs
