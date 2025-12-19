"""
RAST Auxiliary Context
======================

Defines the loop-carried auxiliary context used across RAST iterations.
This module is separate from engine.py to avoid circular imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor


@dataclass
class Aux:
    """Loop-carried auxiliary context for RAST engine.

    This mutable object is passed to Teacher/Assembler each iteration.
    Student does NOT receive Aux (it only processes images).

    Attributes:
        X0:
            - Original observation [B,C,H,W]
        M0:
            - Observation mask (bool or 0/1) [B,1,H,W]
        student_input:
            - Current iteration's input to student [B,C,H,W]
        x_prev:
            - Previous iteration's output [B,C,H,W]
        pipe_meta:
            - Pipeline metadata (scales, noise_sigma, etc.)
        teacher_output_mask:
            - Latest high-confidence mask from teacher [B,1,H,W]
        history_high_conf_masks:
            - History for S-gate temporal consistency
        
        # ===== Hallucination Gate related: Image-centric S-Gate requires historical images =====
        historical_images:
            - History of iteration outputs for S-gate [List[Tensor]]
            - Each element: [B,C,H,W]
        historical_confidence_maps:
            - History of confidence maps for S-gate [List[Tensor]]
            - Each element: [B,1,H,W]
        max_history_length:
            - Maximum number of historical iterations to keep
            - Prevents unbounded memory growth
        
        confidence_threshold:
            - Threshold for high-confidence (for S-gate)
    """
    
    X0: Tensor
    M0: Tensor
    student_input: Tensor  # CRITICAL: current input to student
    x_prev: Optional[Tensor] = None
    pipe_meta: Dict[str, Any] = field(default_factory=dict)
    teacher_output_mask: Optional[Tensor] = None
    
    # ===== Legacy S-Gate field (can be deprecated) =====
    history_high_conf_masks: List[Tensor] = field(default_factory=list)
    
    # ===== Image-centric S-Gate fields =====
    historical_images: List[Tensor] = field(default_factory=list)
    historical_confidence_maps: List[Tensor] = field(default_factory=list)
    max_history_length: int = 10  # Configurable, trade-off between memory and history depth
    
    confidence_threshold: float = 0.5  # For S-gate high-confidence detection