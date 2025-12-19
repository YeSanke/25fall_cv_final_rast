from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn

class Assembler(nn.Module):
    """Composable image assembler for the RAST loop: 
            [x0, M0, student_output_image, M_teacher_t, canvas] -> [final_output_t, canvas]

    Implements the task-agnostic composition rules described by the Engine:
        canvas = canvas * (1.0 - M_teacher) + student_output_image * M_teacher
        final_output_t = x0 * M0 + canvas * (1.0 - M0)
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod # dtype & range guard
    def _as_float01(m: Tensor) -> Tensor:
        if m.dtype == torch.bool:
            m = m.float()
        if not m.dtype.is_floating_point:
            m = (m > 0).float()
        # dtype check
        assert torch.isfinite(m).all(), "mask contains NaN/Inf"
        mn, mx = float(m.min()), float(m.max())
        assert 0.0 - 1e-6 <= mn and mx <= 1.0 + 1e-6, f"mask out of [0,1]: [{mn}, {mx}]"
        return m.clamp_(0.0, 1.0)

    def assemble(
        self,
        *,
        x0: Tensor, # reservation
        M0: Tensor, # mask for reservation
        student_output_image: Tensor,
        M_teacher_t: Tensor,
        canvas: Tensor,
        aux: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor]:
        
        # image shape guard
        assert x0.dim() == student_output_image.dim() == canvas.dim() == 4, "Expect [B,C,H,W]"
        B, C, H, W = x0.shape
        assert student_output_image.shape == (B,C,H,W) and canvas.shape == (B,C,H,W), "shape mismatch (x_cur/canvas_t)"

        # mask shape guard
        assert M0.shape == (B,1,H,W), f"M0 must be [B,1,H,W], got {tuple(M0.shape)}"
        assert M_teacher_t.shape == (B,1,H,W), f"M_teacher_t must be [B,1,H,W], got {tuple(M_teacher_t.shape)}"
        
        M0 = self._as_float01(M0).to(x0)
        M_teacher  = self._as_float01(M_teacher_t).to(x0)
        M_teacher = M_teacher * (1 - M0) # Hard region guard: Ensure the reserved image area

        # canvas update & k-th final output
        canvas = canvas * (1.0 - M_teacher) + student_output_image * M_teacher
        final_output_t = x0 * M0 + canvas * (1.0 - M0)

        return final_output_t, canvas
    
class DenoisingAssembler(Assembler):
    """Image denoising assembler: iteratively replace with cleaner predictions"""
    
    def assemble(self, *, x0, M0, student_output_image, M_teacher_t, canvas, aux=None):
        # Normalize gate mask
        G_t = self._as_float01(M_teacher_t).to(student_output_image)
        
        # Update canvas: trusted regions get new predictions
        canvas_new = canvas * (1.0 - G_t) + student_output_image * G_t
        
        # Output is canvas (no need for M0 in denoising)
        return canvas_new, canvas_new