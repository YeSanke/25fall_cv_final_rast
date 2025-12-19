"""
RAST core — Teacher interface & abstract hallucination gate

This module defines task-agnostic abstractions for the RAST framework:
- Teacher: Dual-head uncertainty estimation (epistemic/aleatoric)
- HallucinationGate: Abstract interface for hallucination detection
- UncertaintyGrader: Abstract interface for uncertainty estimation
- GateConfig: Base configuration dataclass

Task-specific implementations should:
1. Subclass UncertaintyGrader for epistemic/aleatoric graders
2. Subclass HallucinationGate for task-specific hallucination detection
3. Extend GateConfig for additional gate parameters

Variable naming convention:
- student_input: Input TO the student model (e.g., noisy image)
- student_output: Output FROM the student model (e.g., denoised image)
- x_0: Original observation (never changes during RAST loop)
- x_t or x_cur: Current iteration's processed image
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
import abc

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .aux import Aux
from .student import StudentOutputs


# -------------------- Gate configuration (base) --------------------

@dataclass
class GateConfig:
    """Base configuration for hallucination gate.
    
    Task-specific gates should extend this class with additional parameters.
    
    Attributes:
        enabled: Set of enabled sub-gate names (task-specific, e.g., {"D", "S"})
        confidence_threshold: Threshold for high-confidence region in [0,1]
    """
    
    # Enabled sub-gates ("R", "D", "S")
    enabled: set[str] = field(default_factory=set)
    
    # High-confidence threshold
    confidence_threshold: float = 0.5
    

# -------------------- Hallucination Gate (abstract) --------------------

class HallucinationGate(nn.Module, abc.ABC):
    """Abstract base class for task-specific hallucination gates.
    
    Input:
        - student_input: Input to the student model [B,C,H,W]
        - student_output: Output from the student model [B,C,H,W]
        - c_map: Confidence map from teacher [B,1,H,W] in [0,1]
        - aux: Auxiliary context (task-specific, may contain history, evidence, etc.)
        - config: GateConfig instance
    
    Output:
        - G_final: Binary mask [B,1,H,W] in {0,1} indicating trusted regions
        - debug: Dict of intermediate results for visualization/analysis
    
    Subclasses should implement the forward() method with task-specific logic.
    """
    
    def __init__(self, config: Optional[GateConfig] = None):
        """Initialize gate with configuration.
        
        Args:
            config: Gate configuration. If None, will use defaults in forward()
        """
        super().__init__()
        self.config = config
    
    @abc.abstractmethod
    def forward(
        self,
        *,
        student_input: Tensor,
        student_output: Tensor,
        c_map: Tensor,
        aux: Optional[Mapping[str, Any]] = None,
        return_debug: bool = False,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute final trusted mask after hallucination detection.
        
        Args:
            student_input: Input to student model [B,C,H,W]
            student_output: Output from student model [B,C,H,W]
            c_map: Confidence map [B,1,H,W] in [0,1]
            aux: Task-specific auxiliary context
            return_debug: If True, return debug dict with intermediate results
            
        Returns:
            G_final: Binary mask [B,1,H,W] in {0,1}, 1=trusted, 0=rejected
            debug: Dict of intermediate results (empty if return_debug=False)
        """
        raise NotImplementedError


# -------------------- Uncertainty grader abstract class --------------------

class UncertaintyGrader(nn.Module, abc.ABC):
    """Abstract base class for epistemic/aleatoric uncertainty graders.
    
    Graders produce pixel-wise uncertainty maps from student outputs.
    Two types:
    - Epistemic: Model uncertainty (knowledge gaps, out-of-distribution)
    - Aleatoric: Data uncertainty (inherent noise, ambiguity)
    
    Convention: Lower values = higher confidence (lower uncertainty)
    """
    
    @abc.abstractmethod
    def forward(
        self,
        student_output: Tensor,
        *,
        student_input: Optional[Tensor] = None,
        features: Optional[Mapping[str, Tensor]] = None,
        aux: Optional[Mapping[str, Any]] = None,
    ) -> Tensor:
        """Compute uncertainty map.
        
        Args:
            student_output: Output from student model [B,C,H,W]
            student_input: (Optional) Input to student model [B,C,H,W]
            features: (Optional) Intermediate features from student
            aux: (Optional) Auxiliary context
            
        Returns:
            Uncertainty map [B,1,H,W] in [0,1], lower=more confident
        """
        raise NotImplementedError


# -------------------- Teacher --------------------

class Teacher(nn.Module):
    """Abstract Teacher contract for RAST.
    
    Responsibilities:
    1. Estimate epistemic uncertainty (model knowledge gaps)
    2. Estimate aleatoric uncertainty (data noise)
    3. Combine uncertainties into confidence map
    4. Apply hallucination gate to produce final trusted mask
    
    Dataflow:
        student_output -> [e_grader] -> u_e (epistemic uncertainty)
                       -> [a_grader] -> u_a (aleatoric uncertainty)
                       -> [combine]  -> c_map (confidence)
                       -> [h_gate]   -> G_t (trusted mask)
    
    Usage:
        teacher = Teacher(
            e_grader=MCDropoutGrader(...),
            a_grader=VAEGrader(...),
            hallucination_gate=DenoiseHallucinationGate(...)
        )
        
        c_map, G_t = teacher.grading(
            student_output=denoised_image,
            gate_config=config,
            aux=aux_with_student_input_and_history
        )
    """

    name: str = "base-teacher"

    def __init__(
        self,
        *,
        e_grader: Optional[UncertaintyGrader] = None,
        a_grader: Optional[UncertaintyGrader] = None,
        hallucination_gate: Optional[HallucinationGate] = None,
        combine_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    ) -> None:
        """Initialize Teacher.
        
        Args:
            e_grader: Epistemic uncertainty grader (e.g., MC Dropout, ensemble)
            a_grader: Aleatoric uncertainty grader (e.g., VAE, heteroscedastic head)
            hallucination_gate: Task-specific hallucination gate
            combine_fn: Function to combine (u_e, u_a) → c_map
                       Default: geometric mean of (1-u_e) and (1-u_a)
        """
        super().__init__()
        self.e_grader = e_grader
        self.a_grader = a_grader
        self.hallucination_gate = hallucination_gate
        self.combine_fn = combine_fn or self._default_combine
        
    @staticmethod
    def _default_combine(u_e: Tensor, u_a: Tensor) -> Tensor:
        """Default uncertainty combination: geometric mean of confidences.
        
        Formula: c = sqrt((1-u_e) * (1-u_a))
        
        Rationale:
        - Confidence requires BOTH low epistemic and low aleatoric uncertainty
        - Geometric mean is more conservative than arithmetic mean
        - If either uncertainty is high, confidence drops significantly
        
        Args:
            u_e: Epistemic uncertainty [B,1,H,W] in [0,1]
            u_a: Aleatoric uncertainty [B,1,H,W] in [0,1]
            
        Returns:
            Confidence map [B,1,H,W] in [0,1]
        """
        conf_e = (1.0 - u_e).clamp(0, 1)
        conf_a = (1.0 - u_a).clamp(0, 1)
        return torch.sqrt(conf_e * conf_a + 1e-8)


    # ---------  grading  ----------
    def grading(
        self,
        student_output: StudentOutputs,
        aux: Aux,
        *,
        features: Optional[Mapping[str, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Main grading interface called by Engine.
        
        Args:
            student_output: Output from student model [B,C,H,W]
            aux: Auxiliary context from Engine, MUST contain:
                 - student_input: The input that was fed to student [B,C,H,W]
                 - (for gates) history_high_conf_masks, confidence_threshold, etc.
            features: Optional intermediate features from student model
            
        Returns:
            c_map: Combined confidence map [B,1,H,W] in [0,1]
            G_t: Final trusted mask [B,1,H,W] in {0,1}
        
        Raises:
            ValueError: If aux doesn't contain required 'student_input' field
            
        Note:
            Gate configuration should be set when initializing the Teacher
            (via hallucination_gate.__init__(config)), not passed here.
        """
        B, C, H, W = student_output.image.shape
        
        # 1) Compute epistemic uncertainty
        if self.e_grader is not None:
            u_e = self.e_grader(
                student_output.image,
                student_input=getattr(aux, 'student_input', None),
                features=features,
                aux=aux
            )
        else:
            u_e = torch.zeros((B, 1, H, W), device=student_output.image.device, dtype=student_output.image.dtype)
        
        # 2) Compute aleatoric uncertainty
        if self.a_grader is not None:
            u_a = self.a_grader(
                student_output.image,
                student_input=getattr(aux, 'student_input', None),
                features=features,
                aux=aux
            )
        else:
            u_a = torch.zeros((B, 1, H, W), device=student_output.image.device, dtype=student_output.image.dtype)
        
        # 3) Combine into confidence map
        c_map = self.combine_fn(u_e, u_a)
        
        # 4) Apply hallucination gate (if provided)
        if self.hallucination_gate is not None:
            # Config 1
            # ===== H-gate: ENABLED ||  a/e-grader: ENABLED =====
            # CRITICAL: Get student_input from aux
            if not hasattr(aux, 'student_input'):
                raise ValueError(f"[ERROR] aux must contain 'student_input' field for hallucination gate.")
            
            G_t, debug = self.hallucination_gate(
                student_input=aux.student_input,
                student_output=student_output.image,
                c_map=c_map,
                aux=aux,
                return_debug=False,
            )
        else:
            # Config 2 & 3
            # ===== H-gate: DISABLED =====
            # check if graders are enabled
            has_graders = (self.e_grader is not None) or (self.a_grader is not None)
            
            if has_graders:
                # ===== Config 2: Grader ENABLED, filtering with c_map =====
                threshold = getattr(aux, 'confidence_threshold', 0.5)
                G_t = (c_map >= threshold).float()
            else:
                # ===== Config 3: Grader DISABLED，all passed, pure iterations =====
                G_t = torch.ones((B, 1, H, W), device=c_map.device, dtype=c_map.dtype)
        
        return c_map, G_t