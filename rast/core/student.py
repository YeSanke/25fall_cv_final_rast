# student.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

import torch
from torch import Tensor, nn


# ----------------------------
# Dataclass / Abstract class
# ----------------------------
@dataclass
class StudentOutputs:
    """Outputs of Student Module

    Attributes
    ----------
    image: Tensor
        Student output image, shape [B, C, H, W]
    features: Dict[str, Tensor]
        Optional feature maps or embeddings tapped from the backbone
    """

    image: Tensor
    features: Dict[str, Tensor]


# -----------------------------------
# Utilities for robust plug-in design
# -----------------------------------
class Normalizer(nn.Module):
    """Channel-wise affine normalization with optional learnable params.

    If ``learnable=False`` (default), acts as a fixed pre/post transformer.
    ``stats`` may provide keys: 'mean' [1,C,1,1] and 'std' [1,C,1,1].
    If missing, identity is used.
    """

    def __init__(self, channels: int, *, stats: Optional[Mapping[str, Tensor]] = None, learnable: bool = False):
        super().__init__()
        if stats is None:
            mean = torch.zeros(1, channels, 1, 1)
            std = torch.ones(1, channels, 1, 1)
        else:
            mean = stats.get("mean", torch.zeros(1, channels, 1, 1))
            std = stats.get("std", torch.ones(1, channels, 1, 1))
        if learnable:
            self.mean = nn.Parameter(mean)
            self.std = nn.Parameter(std)
        else:
            self.register_buffer("mean", mean, persistent=False)
            self.register_buffer("std", std, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) / (self.std + 1e-8)

    def denorm(self, x: Tensor) -> Tensor:
        return x * (self.std + 1e-8) + self.mean


def _ensure_4d(x: Tensor) -> Tensor:
    if x.dim() == 2:  # [H,W] -> [1,1,H,W]
        return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 3:  # [C,H,W] -> [1,C,H,W]
        return x.unsqueeze(0)
    assert x.dim() == 4, f"Expected 4D tensor [B,C,H,W], got {x.shape}"
    return x


def _maybe_pad_to_divisible(x: Tensor, *, div: int = 16, pad_mode: str = "replicate") -> tuple[Tensor, dict]:
    """Pad spatial dims so H,W are divisible by ``div``.

    Returns (x_pad, info) with info containing 'pad'=(l,r,t,b) for later cropping.
    """
    B, C, H, W = x.shape
    pad_h = (div - (H % div)) % div
    pad_w = (div - (W % div)) % div
    if pad_h == 0 and pad_w == 0:
        return x, {"pad": (0, 0, 0, 0), "div": div}
    # pad: (left, right, top, bottom)
    pad_l = pad_w // 2
    pad_r = pad_w - pad_l
    pad_t = pad_h // 2
    pad_b = pad_h - pad_t
    x_pad = nn.functional.pad(x, (pad_l, pad_r, pad_t, pad_b), mode=pad_mode)
    return x_pad, {"pad": (pad_l, pad_r, pad_t, pad_b), "div": div}


def _crop_to_original(x: Tensor, pad_info: dict) -> Tensor:
    l, r, t, b = pad_info.get("pad", (0, 0, 0, 0))
    if l == r == t == b == 0:
        return x
    return x[..., t : x.shape[-2] - b, l : x.shape[-1] - r]


# --------------------------------------
# Generic wrapper for backbones
# --------------------------------------
class Student(nn.Module):
    """Thin Student wrapper around an arbitrary ``nn.Module`` backbone

    This class DOES NOT perform any pre/post transforms or pad/crop. It assumes
    inputs are already prepared by the student pipeline. Its job is only to:
      - run the backbone under optional AMP
      - optionally collect intermediate layer outputs via hooks
      - optionally transform/curate features via feature_fn
      - return StudentOutputs with image=y, features, and passthrough aux

    Parameters
    ----------
    backbone : 
        nn.Module [B,C,H,W] → [B,C,H,W]
    amp : bool
        If True, run the backbone under autocast mixed precision
    feature_fn : Optional[Callable[[Tensor, Dict[str, Tensor]], Dict[str, Tensor]]]
        Callback invoked after the backbone forward to curate features.
    register_hooks : bool
        If True, attach forward hooks to collect selected layer outputs
    hook_layers : Optional[Dict[str, nn.Module]]
        Named submodules to hook when [register_hooks=True]
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        *,
        amp: bool = False,
        feature_fn: Optional[Callable[[Tensor, Dict[str, Tensor]], Dict[str, Tensor]]]=None,
        register_hooks: bool = False,
        hook_layers: Optional[Dict[str, nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.amp = amp
        self.feature_fn = feature_fn

        self._hook_handles: list[Any] = []
        self._hook_feats: Dict[str, Tensor] = {}
        if register_hooks and hook_layers:
            for name, module in hook_layers.items():
                self._hook_handles.append(
                    module.register_forward_hook(self._make_hook(name))
                )


    # --------------
    # Hook machinery
    # --------------
    def _make_hook(self, name: str):
        def _hook(_m, _inp, out):
            # Store detached views to reduce memory pressure in traces
            if isinstance(out, Tensor):
                self._hook_feats[name] = out
            elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], Tensor):
                self._hook_feats[name] = out[0]
        return _hook

    # ---------
    # Forward
    # ---------
    def forward(self, x_in: Tensor, *, aux: Optional[Dict[str, Any]] = None) -> StudentOutputs:
        aux = {} if aux is None else dict(aux)
        x = _ensure_4d(x_in)

        feats: Dict[str, Tensor] = {}
        self._hook_feats.clear()
        autocast_device = "cuda" if x.is_cuda else "cpu"
        ctx = torch.autocast(autocast_device, enabled=self.amp)
        with ctx:
            y = self.backbone(x)

        if self._hook_feats:
            feats.update({f"hook:{k}": v for k, v in self._hook_feats.items()})
        if self.feature_fn is not None:
            try:
                feats = self.feature_fn(y, feats)
            except Exception as e:
                aux.setdefault("feature_fn_error", str(e))

        return StudentOutputs(image=y, features=feats)