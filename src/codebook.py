import torch
import numpy as np
from typing import Dict, Tuple, List
from typing import List, Optional


class CodeBook:
    """Stores atomic HVs for every possible x, y, t-bucket and polarity p."""

    def __init__(self,
                 dim: int = 8192,
                 seed: int = 0,
                 x_max: int = 35,
                 y_max: int = 35,
                 t_min: int = 0,
                 t_max: int = 10_00_000,
                 t_step: int = 1000
                ):
        self.dim       = dim
        self.gen = torch.Generator().manual_seed(seed)
        self.t_step = t_step

        x = (torch.randn(dim, dtype=torch.float32, generator=self.gen))
        y = (torch.randn(dim, dtype=torch.float32, generator=self.gen))
        t = (torch.randn(dim, dtype=torch.float32, generator=self.gen))
        p = (torch.randn(dim, dtype=torch.float32, generator=self.gen))
        self.x_base = x / x.norm()
        self.y_base = y / y.norm()
        self.t_base = t / t.norm()
        self.p_base = p / p.norm()

        # --- build atom dictionaries -------------------------------------------------
        self.HV_X: Dict[int, torch.tensor] = {
            x: self.fpe(self.x_base, (x+1)/(x_max+1)) for x in range(0, x_max + 1)
        }
        self.HV_Y: Dict[int, torch.tensor] = {
            y: self.fpe(self.y_base, (y+1)/(y_max+1))for y in range(0, y_max + 1)
        }

        # time is quantised into buckets to keep the table moderate in size

        self.HV_T: Dict[int, torch.tensor] = {}
        for t in range(t_min, t_max + 1, t_step):
            bucket = self._t_bucket(t)
            # guarantee one vector per bucket
            if bucket not in self.HV_T:
                self.HV_T[bucket] = self.fpe(self.y_base, bucket/t_max)

        # polarity: 0 (-) / 1 (+)
        self.HV_P: Dict[int, torch.tensor] = {
            0: self.fpe(self.p_base, 0.1),
            1: self.fpe(self.p_base, 0.5)
        }

    # --------------------------------------------------------------------- helpers ---
    def _t_bucket(self, t_us: int) -> int:
        """Quantise a micro-second timestamp to its bucket."""
        return (t_us // self.t_step) * self.t_step

    def bind(self, *hvs: torch.Tensor) -> torch.Tensor:

        out = hvs[0].clone()
        for h in hvs[1:]:
            out = out * h
        return out

    def permute_float(self, hv: torch.Tensor,
                  shift: Optional[int] = None,
                  perm_indices: Optional[torch.Tensor] = None
                 ) -> torch.Tensor:
        if perm_indices is not None:
            # arbitrary permutation
            return hv[perm_indices]
        elif shift is not None:
            # circular permutation
            return torch.roll(hv, shifts=shift, dims=0)
        else:
            raise ValueError("Either `shift` or `perm_indices` must be provided.")

    def fpe(self, base_vector, power):
        """conver hypervector into fourier transform and apply pow. then inverse it"""
        fft_z = torch.fft.fft(base_vector)
        fft_z_frac = torch.pow(fft_z, power)
        result = torch.fft.ifft(fft_z_frac).real
        result = result / torch.linalg.norm(result)
        return result.squeeze(0)

    # ------------------------------------------------------------------- API --------
    def encode_event(self,x: int,y: int,t_us: int,p: int) -> torch.Tensor:
        """Bind X, Y, T, P into one event HV (torch.Tensor)."""
        xb = self.HV_X[x]
        yb = self.HV_Y[y]
        tb = self.HV_T[self._t_bucket(t_us)]
        pb = self.HV_P[p]
        return self.bind(xb, yb, tb, pb)

    def bundle(self, hvs: List[torch.Tensor]) -> torch.Tensor:

        if len(hvs) == 0:
            raise ValueError("Need at least one hyper-vector to bundle")
        # Stack into (N, D) and sum over N
        summed = torch.stack(hvs, dim=0).sum(dim=0)
        # Option A: normalize by length to get unit vector
        return summed / summed.norm(p=2)

    def bundle_events(self,events: List[Tuple[int, int, int, int]]) -> torch.Tensor:
        """Bundle a list of (x,y,t,p) events into one trace HV."""
        return self.bundle([ self.encode_event(*ev) for ev in events ])



