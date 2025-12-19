"""
RAST core — Engine (pipeline loop)

    This module implements the task-agnostic K-step RAST pipeline.

    Architecture:
        Student → Teacher → Assembler (K-loop)

    Responsibilities:
        - Orchestrate the recurrent loop
        - Maintain loop-carried context (Aux)
        - Collect optional traces for analysis
        - DOES NOT know about gate/grader internals

    Key Design Principles:
        - Engine is completely task-agnostic
        - All task-specific logic is encapsulated in Student/Teacher/Assembler
        - Configuration happens at component initialization, not in Engine
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List

import torch
from torch import Tensor, nn

from .student_pipeline import StudentPipeline
from .teacher import Teacher
from .assembler import Assembler
from .aux import Aux


# ------------------------- Engine -------------------------

class RASTEngine(nn.Module):
    """Task-agnostic RAST pipeline engine.
        
        Orchestrates: Student → Teacher → Assembler for K iterations.
        
        Workflow (per iteration):
            1. Student: student_input → student_output
            2. Update aux.student_input for Teacher
            3. Teacher: (student_output, aux) → (c_map, G_t)
            4. Update history in aux
            5. Assembler: (X0, student_output, G_t, canvas, aux) → x_next
            6. Loop: student_input = x_next for next iteration
        
        Usage:
        [1]    engine = RASTEngine(
                            student=student_pipeline,
                            teacher=teacher_with_configured_gate,
                            assembler=assembler,
                            steps=5
                        )
            
        [2]    aux = Aux(X0=noisy_image, M0=mask, student_input=noisy_image)
        [3]    x_final, traces = engine.run(aux, return_traces=True)
    """

    def __init__(
        self,
        student: StudentPipeline,
        teacher: Teacher,
        assembler: Assembler,
        *,
        steps: int = 5,
    ) -> None:
        """Initialize RAST engine.
        
        Args:
            student: Student pipeline (already wrapped with pre/post ops)
            teacher: Teacher with configured graders and hallucination gate
            assembler: Assembler for composing final output
            steps: Number of RAST iterations
        
        Note:
            All configuration (gate thresholds, grader params, etc.) should
            be done when initializing the components, NOT passed to Engine.
        """
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.assembler = assembler
        self.steps = int(steps)
        
    
    # -------------------- Helper --------------------
    def _update_sgate_history(
            self, 
            aux: Aux, 
            current_output: Tensor, 
            current_confidence: Tensor
        ):
            """
            Update S-Gate history in aux after each iteration.
            
            This method is called by Engine after teacher grading but before assembler.
            It maintains a sliding window of historical outputs and confidence maps.
            
            Args:
                aux: Auxiliary context (modified in-place)
                current_output: Current iteration's student output [B,C,H,W]
                current_confidence: Current iteration's confidence map [B,1,H,W]
            """
            # Detach tensors to prevent gradient tracking through history
            current_output_detached = current_output.detach()
            current_confidence_detached = current_confidence.detach()
            
            # Optional: Apply memory optimization (if configured in aux)
            if hasattr(aux, 'use_half_precision_history') and aux.use_half_precision_history:
                current_output_detached = current_output_detached.half()
                current_confidence_detached = current_confidence_detached.half()
            
            if hasattr(aux, 'history_downsample_factor') and aux.history_downsample_factor < 1.0:
                import torch.nn.functional as F
                current_output_detached = F.interpolate(
                    current_output_detached,
                    scale_factor=aux.history_downsample_factor,
                    mode='bilinear',
                    align_corners=False
                )
                current_confidence_detached = F.interpolate(
                    current_confidence_detached,
                    scale_factor=aux.history_downsample_factor,
                    mode='bilinear',
                    align_corners=False
                )
            
            # Append to history
            aux.historical_images.append(current_output_detached)
            aux.historical_confidence_maps.append(current_confidence_detached)
            
            # Maintain sliding window (prevent unbounded memory growth)
            max_len = aux.max_history_length
            if len(aux.historical_images) > max_len:
                aux.historical_images.pop(0)
                aux.historical_confidence_maps.pop(0)
            
            # Update legacy field for backward compatibility
            if hasattr(aux, 'history_high_conf_masks'):
                high_conf_mask = (current_confidence_detached >= aux.confidence_threshold).float()
                aux.history_high_conf_masks.append(high_conf_mask)
                if len(aux.history_high_conf_masks) > max_len:
                    aux.history_high_conf_masks.pop(0)
    

    # -------------------- Main Loop --------------------
    @torch.no_grad()
    def run(
        self,
        aux: Aux,
        *,
        return_traces: bool = False
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Run RAST loop for configured number of steps.
        
        Args:
            aux: Loop-carried context containing X0, M0, student_input, etc.
            return_traces: If True, collect intermediate results for analysis
            
        Returns:
            x_final: Final output after K iterations [B,C,H,W]
            traces: Dict with lists of intermediate results:
                    - "student_outputs": List of student outputs per iteration
                    - "confidence_maps": List of teacher confidence maps
                    - "gate_masks": List of final gate masks
                    - (empty dict if return_traces=False)
        """
        device = aux.X0.device
        canvas = aux.student_input.clone().to(device)  # Start from noisy image

        # Initialize traces
        traces: Dict[str, List[Tensor]] = {}
        if return_traces:
            traces = {
                "student_outputs": [],
                "confidence_maps": [],
                "gate_masks": [],
            }

        for t in range(self.steps):
            # ===== Step 1: Student Forward =====
            # Student only sees images, not aux
            student_input_t = aux.student_input.to(device)
            student_output_t = self.student.forward(student_input_t)

            # ===== Step 2: Update aux for Teacher =====
            # CRITICAL: Teacher needs to know what student received as input
            # (already set in aux.student_input, but ensure it's current)
            aux.student_input = student_input_t

            # ===== Step 3: Teacher Grading =====
            # Teacher uses:
            # - student_output_t for uncertainty estimation
            # - aux.student_input for hallucination gate (e.g., D-gate residual check)
            # - aux.history_high_conf_masks for S-gate temporal consistency
            # - aux.historical_images for image-centric S-gate
            # - aux.historical_confidence_maps for image-centric S-gate
            c_map, G_t = self.teacher.grading(
                student_output=student_output_t,
                aux=aux,
            )
            
            # ===== Step 4: Update aux with Teacher outputs =====
            aux.teacher_output_mask = G_t
            aux.x_prev = student_output_t.image
            
            # ===== Step 4.5: Update S-Gate History =====
            # CRITICAL: Must happen BEFORE assembler, AFTER teacher grading
            # Update historical images and confidence maps for next iteration's S-gate
            self._update_sgate_history(
                aux=aux,
                current_output=student_output_t.image,  # Current iteration's output
                current_confidence=c_map,          # Current iteration's confidence
            )

            # ===== Step 5: Assembler =====
            # Compose next iteration's input
            x_next, canvas = self.assembler.assemble(
                x0=aux.X0,
                M0=aux.M0,
                student_output_image=student_output_t.image,
                M_teacher_t=G_t,
                canvas=canvas,
                aux=aux
            )

            # ===== Step 6: Loop Update =====
            aux.student_input = x_next  # Next iteration's student input

            # ===== Step 7: Collect Traces (optional) =====
            if return_traces:
                traces["student_outputs"].append(student_output_t.image.detach().cpu())
                traces["confidence_maps"].append(c_map.detach().cpu())
                traces["gate_masks"].append(G_t.detach().cpu())

        # Final output is the last canvas
        x_final = canvas
        return x_final, traces

