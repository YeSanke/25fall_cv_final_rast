from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

import torch
from torch import Tensor, nn

from rast.core.student import Student, StudentOutputs
from rast.core.student import _ensure_4d, _maybe_pad_to_divisible, _crop_to_original

class StudentPipeline(nn.Module):
    """A image2image student pipeline for external training/inference. module and pre/proc
    process in student.py is used to build the pipeline

        s_input -> [preproc] -> [backbone] -> [postproc] -> [crop back] -> s_output
    """

    def __init__(
        self,
        backbone: Student,
        *,
        preproc: Optional[nn.Module] = None,
        postproc: Optional[nn.Module] = None,
        pad_divisible: int = 16,
        amp: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.preproc = preproc
        self.postproc = postproc
        self.pad_divisible = int(pad_divisible)
        self.amp = amp

    def forward(self, s_input: Tensor, *, aux: Optional[Dict[str, Any]] = None) -> StudentOutputs:
        s_input = _ensure_4d(s_input)
        
        # pre
        if self.preproc is not None:
            s_after_pre, pad_info = _maybe_pad_to_divisible(...)
            s_after_pre = self.preproc(s_after_pre)
        else:
            s_after_pre, pad_info = _maybe_pad_to_divisible(s_input, div=self.pad_divisible)
            
        # backbone
        autocast_device = "cuda" if s_after_pre.is_cuda else "cpu"
        with torch.autocast(autocast_device, enabled=self.amp):
            student_outputs = self.backbone.forward(s_after_pre, aux=aux)
            s_after_backbone = student_outputs.image
            
        # proc
        if self.postproc is not None:
            s_after_post = self.postproc(s_after_backbone)
        else:
            s_after_post = s_after_backbone
        s_output = _crop_to_original(s_after_post, pad_info)
        return StudentOutputs(image=s_output, features=student_outputs.features)