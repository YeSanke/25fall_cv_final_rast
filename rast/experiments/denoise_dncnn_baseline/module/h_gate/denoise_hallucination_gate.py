"""
Denoising-specific hallucination gate implementation

Implements hybrid soft-hard hallucination detection for image denoising tasks.

Gate Strategy:
- D-Gate: Detects residual anomalies (non-noise patterns in student_output - student_input)
- S-Gate: Detects temporal inconsistency across RAST iterations (IMAGE-CENTRIC, KERNEL-BASED)

Three-stage filtering:
1. Hard Reject: Top-k% worst pixels per gate -> directly rejected
2. Soft Fusion: Remaining pixels -> soft aggregation -> continuous scores
3. Final Selection: Top-p% from soft scores -> final binary mask
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from core.teacher import GateConfig, HallucinationGate


# -------------------- Denoising Gate Configuration --------------------

@dataclass
class DenoiseGateConfig(GateConfig):
    """Configuration for denoising hallucination gate.
    
    Extends base GateConfig with denoising-specific parameters.
    """
    
    # Enabled sub-gates for denoising (typically {"D", "S"})
    enabled: set[str] = field(default_factory=lambda: {"D", "S"})
    
    # ===== Stage 1: Hard Reject =====
    hard_reject_ratio: float = 0.2  # Reject worst 20% per gate
    
    # ===== Stage 2: Soft Fusion =====
    soft_aggregation: str = "product"  # 'product' | 'weighted' | 'min'
    gate_steepness: float = 10.0       # Sigmoid steepness for soft gates
    
    # Soft gate thresholds (for sigmoid centering)
    theta_D: float = 0.5  # D-gate threshold
    theta_S: float = 0.5  # S-gate threshold
    
    # Weighted aggregation (only for soft_aggregation='weighted')
    weight_D: float = 0.6  # D-gate weight (residual check more reliable)
    weight_S: float = 0.4  # S-gate weight
    
    # ===== Stage 3: Final Selection =====
    soft_selection_mode: str = "top_k"  # 'threshold' | 'top_k' | 'adaptive'
    
    # Method A: threshold mode
    soft_threshold: float = 0.5
    
    # Method B: top_k mode
    soft_keep_ratio: float = 0.5  # Keep top 50% from soft fusion
    
    # Method C: adaptive mode
    adaptive_std_factor: float = 0.5  # threshold = mean + factor*std
    
    # ===== D-Gate: Residual anomaly detection weights =====
    d_gate_var_weight: float = 0.4    # Local variance feature weight
    d_gate_freq_weight: float = 0.3   # High-frequency energy weight
    d_gate_grad_weight: float = 0.3   # Gradient smoothness weight
    
    # ===== S-Gate: Temporal consistency parameters (IMAGE-CENTRIC, KERNEL-BASED) =====
    # Iteration stage control
    s_gate_warmup_iterations: int = 3          # First N iterations: S-Gate completely off
    s_gate_enable_iterations: int = 5          # Nth iteration: S-Gate at full strength
    
    # Kernel-based evaluation with adaptive receptive field
    s_gate_initial_kernel_size: int = 11       # Early: large receptive field (global view)
    s_gate_final_kernel_size: int = 3          # Late: small receptive field (detail focus)
    s_gate_kernel_decay_rate: float = 0.8      # Receptive field shrink rate
    
    # Two-branch weights (simplified from three)
    s_gate_gradient_weight: float = 0.6        # Gradient consistency weight
    s_gate_semantic_weight: float = 0.4        # Semantic stability weight
    
    # History window
    s_gate_history_window: int = 10            # Max history length to keep


# -------------------- D-Gate: Residual Anomaly Detection --------------------

def d_gate_residual_anomaly(
    student_input: Tensor,
    student_output: Tensor,
    c_map: Tensor,
    aux: Optional[Mapping[str, Any]] = None,
    *,
    var_weight: float = 0.4,
    freq_weight: float = 0.3,
    grad_weight: float = 0.3,
) -> Tensor:
    """
    D-Gate: Detect anomalies in denoising residuals.
    
    Intuition:
        residual = student_output - student_input should resemble "real noise"
        
    Anomaly indicators:
        1. Low local variance -> structured patterns (not random noise)
        2. Concentrated high-frequency energy -> periodic artifacts
        3. Low gradient variance -> smooth/coherent structures
    
    Args:
        student_input: Input noisy image [B,C,H,W]
        student_output: Denoised output [B,C,H,W]
        c_map: Confidence map [B,1,H,W] (not used here, kept for interface)
        aux: Auxiliary context (optional)
        var_weight: Weight for local variance feature
        freq_weight: Weight for frequency feature
        grad_weight: Weight for gradient feature
        
    Returns:
        violation_score [B,1,H,W] in [0,1], higher = more suspicious
    """
    residual = student_output - student_input
    B, C, H, W = residual.shape
    
    kernel_size = 7
    pad = kernel_size // 2
    
    # ============ Feature 1: Local Variance ============
    # True noise: high local variance (random fluctuations)
    # Artifacts: low local variance (coherent structures)
    
    residual_padded = F.pad(residual, (pad, pad, pad, pad), mode='reflect')
    unfold = F.unfold(residual_padded, kernel_size, stride=1)  # [B, C*k*k, H*W]
    
    # Local mean and variance
    local_mean = unfold.mean(dim=1, keepdim=True)
    local_var = ((unfold - local_mean) ** 2).mean(dim=1, keepdim=True)
    local_var = local_var.reshape(B, 1, H, W)
    
    # Normalize and invert: low variance → high score (suspicious)
    var_norm = (local_var - local_var.min()) / (local_var.max() - local_var.min() + 1e-6)
    var_score = 1.0 - var_norm
    
    # ============ Feature 2: High-Frequency Energy ============
    # True noise: flat power spectrum (white noise property)
    # Artifacts: energy concentrated at specific frequencies
    
    residual_fft = torch.fft.rfft2(residual)
    power_spectrum = torch.abs(residual_fft) ** 2
    
    # Define high-frequency region (simple: lower half of frequency domain)
    _, _, fH, fW = power_spectrum.shape
    high_freq_mask = torch.zeros_like(power_spectrum)
    high_freq_mask[:, :, fH // 2:, :] = 1.0
    
    # Energy ratio
    high_freq_energy = (power_spectrum * high_freq_mask).sum(dim=[2, 3], keepdim=True)
    total_energy = power_spectrum.sum(dim=[2, 3], keepdim=True) + 1e-6
    freq_ratio = (high_freq_energy / total_energy).expand(-1, -1, H, W)
    
    # Normalize: deviation from expected ratio indicates anomaly
    freq_score = (freq_ratio - freq_ratio.min()) / (freq_ratio.max() - freq_ratio.min() + 1e-6)
    
    # ============ Feature 3: Gradient Smoothness ============
    # True noise: chaotic gradients (high local std)
    # Artifacts: smooth gradients (low local std, directional coherence)
    
    grad_x = torch.abs(residual[:, :, 1:, :] - residual[:, :, :-1, :])
    grad_y = torch.abs(residual[:, :, :, 1:] - residual[:, :, :, :-1])
    
    # Pad to original size
    grad_x_pad = F.pad(grad_x, (0, 0, 0, 1))
    grad_y_pad = F.pad(grad_y, (0, 1, 0, 0))
    grad_total = grad_x_pad + grad_y_pad
    
    # Local gradient standard deviation
    grad_padded = F.pad(grad_total, (pad, pad, pad, pad), mode='reflect')
    grad_unfold = F.unfold(grad_padded, kernel_size, stride=1)
    grad_std = grad_unfold.std(dim=1, keepdim=True).reshape(B, 1, H, W)
    
    # Normalize and invert: low std → high score (smooth = suspicious)
    grad_score = 1.0 - (grad_std / (grad_std.max() + 1e-6))
    
    # ============ Combine Features ============
    violation_score = (
        var_weight * var_score +
        freq_weight * freq_score +
        grad_weight * grad_score
    )
    
    # Average across channels if multi-channel
    return violation_score.mean(dim=1, keepdim=True)  # [B,1,H,W]


# -------------------- S-Gate: Temporal Consistency (IMAGE-CENTRIC, KERNEL-BASED) --------------------

def _get_adaptive_kernel_size(
    num_iterations: int,
    initial_size: int = 11,
    final_size: int = 3,
    decay_rate: float = 0.8,
) -> int:
    """
    Calculate adaptive kernel size based on iteration number.
    
    As iterations progress, receptive field gradually shrinks:
    - Early iterations: Large kernel (11x11) → Focus on global structure
    - Late iterations: Small kernel (3x3) → Focus on local details
    
    Intuition:
        Early stage: Ensure overall direction is correct (far view)
        Late stage: Refine texture details (close view)
    
    Args:
        num_iterations: Current iteration count (after warmup)
        initial_size: Initial kernel size
        final_size: Final kernel size
        decay_rate: Decay rate (smaller = faster decay)
        
    Returns:
        kernel_size: Odd kernel size
    """
    # Exponential decay: size = final + (initial - final) * decay^iter
    size = final_size + (initial_size - final_size) * (decay_rate ** num_iterations)
    
    # Ensure odd number
    kernel_size = int(size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Clamp to range
    kernel_size = max(final_size, min(initial_size, kernel_size))
    
    return kernel_size


def _compute_kernel_based_gradient_consistency(
    current_image: Tensor,
    historical_images: list[Tensor],
    historical_confidence_maps: list[Tensor],
    kernel_size: int,
) -> Tensor:
    """
    Branch 1: Kernel-based Gradient Consistency Detection.
    
    Instead of pixel-wise computation, evaluate within local neighborhoods:
    - Compute local average change vectors
    - Evaluate consistency of local change directions
    
    Intuition:
        Allow pixel-level drastic changes (texture details),
        but require local neighborhood's overall direction to remain consistent.
        
    This solves the texture-rich image problem: even when processing fine details
    correctly, pixel values may change drastically, but the local trend should be stable.
    
    Args:
        current_image: Current iteration output [B,C,H,W]
        historical_images: List of past iteration outputs, each [B,C,H,W]
        historical_confidence_maps: List of past confidence maps, each [B,1,H,W]
        kernel_size: Receptive field size for local averaging
        
    Returns:
        violation_score [B,1,H,W] in [0,1], higher = more inconsistent
    """
    if len(historical_images) < 2:
        return torch.zeros(
            current_image.shape[0], 1,
            current_image.shape[2], current_image.shape[3],
            device=current_image.device, dtype=current_image.dtype
        )
    
    prev_image = historical_images[-1]       # t
    prev_prev_image = historical_images[-2]  # t-1
    prev_conf = historical_confidence_maps[-1]  # Confidence at t
    
    B, C, H, W = current_image.shape
    pad = kernel_size // 2
    
    # Compute change vectors [B, C, H, W]
    delta_prev = prev_image - prev_prev_image      # Δ(t-1→t)
    delta_curr = current_image - prev_image        # Δ(t→t+1)
    
    # ===== KEY: Use average pooling to obtain local average changes =====
    # This smooths out texture-level pixel variations, focusing on overall trends
    
    delta_prev_smooth = F.avg_pool2d(
        F.pad(delta_prev, (pad, pad, pad, pad), mode='reflect'),
        kernel_size=kernel_size,
        stride=1,
        padding=0
    )  # [B, C, H, W]
    
    delta_curr_smooth = F.avg_pool2d(
        F.pad(delta_curr, (pad, pad, pad, pad), mode='reflect'),
        kernel_size=kernel_size,
        stride=1,
        padding=0
    )  # [B, C, H, W]
    
    # Now we compute consistency of "average change directions in local neighborhoods"
    delta_prev_flat = delta_prev_smooth.reshape(B, C, H * W)
    delta_curr_flat = delta_curr_smooth.reshape(B, C, H * W)
    
    # Cosine similarity: treat C-dimensional vectors at each spatial location
    dot_product = (delta_prev_flat * delta_curr_flat).sum(dim=1)  # [B, H*W]
    norm_prev = torch.norm(delta_prev_flat, dim=1) + 1e-8         # [B, H*W]
    norm_curr = torch.norm(delta_curr_flat, dim=1) + 1e-8         # [B, H*W]
    
    cosine_sim = dot_product / (norm_prev * norm_curr)  # [B, H*W], range [-1, 1]
    cosine_sim = cosine_sim.reshape(B, 1, H, W)         # [B, 1, H, W]
    
    # Convert to inconsistency score
    # cosine_sim = -1 (opposite direction) → inconsistency = 1.0
    # cosine_sim =  0 (orthogonal)        → inconsistency = 0.5
    # cosine_sim = +1 (same direction)    → inconsistency = 0.0
    inconsistency = (1.0 - cosine_sim) / 2.0  # Map [-1,1] to [0,1]
    
    # Weight by historical confidence (also use local average)
    # High confidence regions: direction changes are more suspicious
    # Low confidence regions: allow exploration in different directions
    prev_conf_smooth = F.avg_pool2d(
        F.pad(prev_conf, (pad, pad, pad, pad), mode='reflect'),
        kernel_size=kernel_size,
        stride=1,
        padding=0
    )
    
    gradient_violation = inconsistency * prev_conf_smooth
    
    return gradient_violation


def _compute_kernel_based_semantic_stability(
    current_image: Tensor,
    historical_images: list[Tensor],
    historical_confidence_maps: list[Tensor],
    kernel_size: int,
    window_size: int = 5,
) -> Tensor:
    """
    Branch 2: Kernel-based Semantic Stability Detection.
    
    Evaluate whether "semantic content" of local regions is stable:
    - Don't look at pixel values, look at local statistics (mean, variance)
    - Local regions with high confidence should have stable statistical features
    
    Intuition:
        The "general appearance" of a local patch should be stable,
        but allow internal texture details to continue optimizing.
        
    This allows pixel-level refinement while ensuring the patch's semantic
    identity (e.g., "smooth sky" vs "textured grass") doesn't flip-flop.
    
    Args:
        current_image: Current iteration output [B,C,H,W]
        historical_images: List of past iteration outputs
        historical_confidence_maps: List of past confidence maps
        kernel_size: Receptive field size for local feature extraction
        window_size: Number of recent iterations to consider
        
    Returns:
        violation_score [B,1,H,W] in [0,1], higher = more suspicious
    """
    if len(historical_images) < 3:
        return torch.zeros(
            current_image.shape[0], 1,
            current_image.shape[2], current_image.shape[3],
            device=current_image.device, dtype=current_image.dtype
        )
    
    recent_history = historical_images[-window_size:]
    recent_confs = historical_confidence_maps[-window_size:]
    
    B, C, H, W = current_image.shape
    pad = kernel_size // 2
    
    # ===== Extract Local Statistical Features =====
    # Compute mean and std for each patch as "semantic representation"
    
    def extract_local_features(img):
        """Extract local mean and standard deviation"""
        # Local mean
        local_mean = F.avg_pool2d(
            F.pad(img, (pad, pad, pad, pad), mode='reflect'),
            kernel_size=kernel_size,
            stride=1,
            padding=0
        )  # [B, C, H, W]
        
        # Local standard deviation
        img_padded = F.pad(img, (pad, pad, pad, pad), mode='reflect')
        img_unfold = F.unfold(img_padded, kernel_size, stride=1)  # [B, C*K*K, H*W]
        local_std = img_unfold.std(dim=1, keepdim=True).reshape(B, 1, H, W)
        
        # Concatenate mean and std as features
        # Expand std to match channels for concatenation
        return torch.cat([local_mean, local_std.expand(-1, C, -1, -1)], dim=1)  # [B, 2*C, H, W]
    
    # Extract local features for history and current
    history_features = [extract_local_features(img) for img in recent_history]
    current_features = extract_local_features(current_image)
    
    # Compute variance of historical features (stability in feature space)
    history_stack = torch.stack(history_features, dim=0)  # [T, B, 2*C, H, W]
    feature_variance = torch.var(history_stack, dim=0)  # [B, 2*C, H, W]
    feature_variance_norm = feature_variance / (feature_variance.max() + 1e-6)
    
    # Stability: low feature variance = semantically stable
    stability = torch.exp(-feature_variance_norm * 3)
    
    # Historical average confidence (local average)
    avg_conf = torch.stack(recent_confs, dim=0).mean(dim=0)
    avg_conf_smooth = F.avg_pool2d(
        F.pad(avg_conf, (pad, pad, pad, pad), mode='reflect'),
        kernel_size=kernel_size,
        stride=1,
        padding=0
    )
    
    # Current feature change
    feature_change = torch.abs(current_features - history_features[-1])
    feature_change_norm = feature_change / (feature_change.max() + 1e-6)
    
    # Penalty: high confidence × stable features × large feature change
    penalty = stability * avg_conf_smooth * feature_change_norm
    
    violation = torch.sigmoid(penalty.mean(dim=1, keepdim=True) * 10 - 5)
    
    return violation  # [B, 1, H, W]


def s_gate_temporal_consistency(
    student_input: Tensor,
    student_output: Tensor,
    c_map: Tensor,
    aux: Optional[Mapping[str, Any]] = None,
    *,
    gradient_weight: float = 0.6,
    semantic_weight: float = 0.4,
    warmup_iterations: int = 3,
    enable_iterations: int = 5,
    initial_kernel_size: int = 11,
    final_kernel_size: int = 3,
    kernel_decay_rate: float = 0.8,
    history_window: int = 10,
) -> Tensor:
    """
    S-Gate: Simplified two-branch temporal consistency detection.
    
    Key Features:
    1. Early shutdown: First warmup_iterations are completely disabled
    2. Kernel-based: Uses adaptive receptive field (early: global, late: local)
    3. Two branches: Gradient consistency + Semantic stability
    
    Design Philosophy:
        - Allow free exploration in early iterations (warmup)
        - Evaluate local neighborhoods, not individual pixels (kernel-based)
        - As iterations progress, focus shifts from global to local (adaptive receptive field)
        - Gradually ramp up constraint strength (smooth transition)
    
    Args:
        student_input: Input [B,C,H,W] (unused, kept for interface)
        student_output: Current iteration output [B,C,H,W]
        c_map: Current confidence map [B,1,H,W]
        aux: Must contain:
            - 'historical_images': List[Tensor] of past iteration outputs
            - 'historical_confidence_maps': List[Tensor] of past confidence maps
        gradient_weight: Weight for gradient consistency branch
        semantic_weight: Weight for semantic stability branch
        warmup_iterations: First N iterations completely disabled
        enable_iterations: Nth iteration reaches full strength
        initial_kernel_size: Initial receptive field size
        final_kernel_size: Final receptive field size
        kernel_decay_rate: Receptive field decay rate
        history_window: Maximum history length to keep
        
    Returns:
        violation_score [B,1,H,W] in [0,1], higher = more suspicious
    """
    if aux is None:
        raise ValueError("S-gate requires 'aux' with history information")
    
    history_imgs = getattr(aux, 'historical_images', [])
    history_confs = getattr(aux, 'historical_confidence_maps', [])
    
    num_iterations = len(history_imgs)
    
    # ===== Stage 1: Warmup Phase (Complete Shutdown) =====
    if num_iterations < warmup_iterations:
        return torch.zeros_like(c_map)
    
    # ===== Stage 2: Compute Adaptive Kernel Size =====
    iterations_since_warmup = num_iterations - warmup_iterations
    kernel_size = _get_adaptive_kernel_size(
        iterations_since_warmup,
        initial_kernel_size,
        final_kernel_size,
        kernel_decay_rate
    )
    
    # Limit history window
    if len(history_imgs) > history_window:
        history_imgs = history_imgs[-history_window:]
        history_confs = history_confs[-history_window:]
    
    # ===== Branch 1: Kernel-based Gradient Consistency =====
    gradient_violation = _compute_kernel_based_gradient_consistency(
        student_output, history_imgs, history_confs, kernel_size
    )
    
    # ===== Branch 2: Kernel-based Semantic Stability =====
    semantic_violation = _compute_kernel_based_semantic_stability(
        student_output, history_imgs, history_confs, kernel_size, history_window
    )
    
    # ===== Weighted Fusion =====
    total_violation = (
        gradient_weight * gradient_violation +
        semantic_weight * semantic_violation
    )
    
    # ===== Stage 3: Progressive Ramp-up =====
    # After warmup, gradually increase strength to full
    if num_iterations < enable_iterations:
        ramp_factor = (num_iterations - warmup_iterations) / (enable_iterations - warmup_iterations)
        total_violation = total_violation * ramp_factor
    
    return total_violation.clamp(0.0, 1.0)


# -------------------- Denoising Hallucination Gate --------------------

class DenoiseHallucinationGate(HallucinationGate):
    """
    Hallucination gate for image denoising tasks.
    
    Implements hybrid soft-hard filtering with D-Gate and S-Gate:
    
    Stage 1 (Hard Reject):
        - Each gate identifies its worst top-k% pixels
        - These pixels are unconditionally rejected (safety mechanism)
        
    Stage 2 (Soft Fusion):
        - Remaining pixels get soft scores via sigmoid gates
        - Scores are aggregated (product/weighted/min)
        - Produces continuous confidence in [0,1]
        
    Stage 3 (Final Selection):
        - Select top-p% from soft scores (threshold/top_k/adaptive)
        - Produces final binary mask {0,1}
    
    Usage:
        gate = DenoiseHallucinationGate(config)
        G_final, debug = gate(
            student_input=x_noisy,
            student_output=x_denoised,
            c_map=confidence,
            aux=aux_with_history,
        )
    """
    
    def __init__(self, config: Optional[DenoiseGateConfig] = None):
        """Initialize denoising hallucination gate.
        
        Args:
            config: Configuration for gate behavior. If None, uses defaults.
        """
        super().__init__(config)
        if self.config is None:
            self.config = DenoiseGateConfig()
        elif not isinstance(self.config, DenoiseGateConfig):
            # Convert base GateConfig to DenoiseGateConfig
            self.config = DenoiseGateConfig(
                enabled=self.config.enabled,
                confidence_threshold=self.config.confidence_threshold,
            )
    
    def forward(
        self,
        *,
        student_input: Tensor,
        student_output: Tensor,
        c_map: Tensor,
        aux: Optional[Mapping[str, Any]] = None,
        config: Optional[DenoiseGateConfig] = None,
        return_debug: bool = False,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Apply hybrid soft-hard hallucination gate.
        
        Args:
            student_input: Input noisy image [B,C,H,W]
            student_output: Denoised output [B,C,H,W]
            c_map: Confidence map [B,1,H,W] in [0,1]
            aux: Auxiliary context (must contain history for S-gate)
            config: Override default config for this call
            return_debug: If True, return detailed debug info
            
        Returns:
            G_final: Binary trusted mask [B,1,H,W] in {0,1}
            debug: Dict of intermediate results (empty if return_debug=False)
        """
        cfg: DenoiseGateConfig = config or self.config  # type: ignore
        B, _, H, W = c_map.shape
        k = cfg.gate_steepness
        
        debug: Dict[str, Tensor] = {}
        
        # ========== Stage 0: High-Confidence Region ==========
        high_conf = (c_map >= cfg.confidence_threshold)
        debug["high_conf"] = high_conf.float()
        
        # ========== Stage 1: Hard Reject ==========
        hard_reject_mask = torch.zeros_like(high_conf, dtype=torch.bool)
        enabled = set(cfg.enabled or set())
        
        # D-Gate hard reject
        if "D" in enabled:
            d_score = d_gate_residual_anomaly(
                student_input, student_output, c_map, aux,
                var_weight=cfg.d_gate_var_weight,
                freq_weight=cfg.d_gate_freq_weight,
                grad_weight=cfg.d_gate_grad_weight,
            )
            debug["d_score"] = d_score
            
            # Compute percentile only within high-confidence region
            d_score_masked = torch.where(
                high_conf, d_score, torch.full_like(d_score, -1e9)
            )
            
            d_flat = d_score_masked.reshape(B, -1)
            for i in range(B):
                n_high_conf = high_conf[i].sum()
                if n_high_conf > 0:
                    # Find (1 - hard_reject_ratio) percentile
                    threshold = torch.quantile(
                        d_flat[i][d_flat[i] > -1e8],
                        1.0 - cfg.hard_reject_ratio,
                        interpolation='higher'
                    )
                    hard_reject_mask[i] |= (d_score[i] > threshold) & high_conf[i]
            
            debug["d_hard_reject"] = hard_reject_mask.float()
        
        # S-Gate hard reject
        if "S" in enabled:
            s_score = s_gate_temporal_consistency(
                student_input, student_output, c_map, aux,
                gradient_weight=cfg.s_gate_gradient_weight,
                semantic_weight=cfg.s_gate_semantic_weight,
                warmup_iterations=cfg.s_gate_warmup_iterations,
                enable_iterations=cfg.s_gate_enable_iterations,
                initial_kernel_size=cfg.s_gate_initial_kernel_size,
                final_kernel_size=cfg.s_gate_final_kernel_size,
                kernel_decay_rate=cfg.s_gate_kernel_decay_rate,
                history_window=cfg.s_gate_history_window,
            )
            debug["s_score"] = s_score
            
            # Compute percentile only within high-confidence region
            s_score_masked = torch.where(
                high_conf, s_score, torch.full_like(s_score, -1e9)
            )
            
            s_flat = s_score_masked.reshape(B, -1)
            for i in range(B):
                n_high_conf = high_conf[i].sum()
                if n_high_conf > 0:
                    threshold = torch.quantile(
                        s_flat[i][s_flat[i] > -1e8],
                        1.0 - cfg.hard_reject_ratio,
                        interpolation='higher'
                    )
                    hard_reject_mask[i] |= (s_score[i] > threshold) & high_conf[i]
            
                debug["s_hard_reject"] = hard_reject_mask.float()
        
        debug["hard_reject_total"] = hard_reject_mask.float()
        
        # ========== Stage 2: Soft Fusion ==========
        # Soft confidence gate (sigmoid around threshold)
        g_conf = torch.sigmoid(k * (c_map - cfg.confidence_threshold))
        
        # Compute soft penalties for each gate
        if "D" in enabled:
            penalty_D = torch.sigmoid(k * (d_score - cfg.theta_D))
            debug["penalty_D"] = penalty_D
        else:
            penalty_D = torch.zeros_like(g_conf)
        
        if "S" in enabled:
            penalty_S = torch.sigmoid(k * (s_score - cfg.theta_S))
            debug["penalty_S"] = penalty_S
        else:
            penalty_S = torch.zeros_like(g_conf)
        
        # Aggregate soft scores
        if cfg.soft_aggregation == "product":
            # Multiplicative: any high penalty significantly reduces score
            G_soft = g_conf * (1.0 - penalty_D) * (1.0 - penalty_S)
            
        elif cfg.soft_aggregation == "weighted":
            # Weighted sum: allows tuning relative importance
            combined_penalty = cfg.weight_D * penalty_D + cfg.weight_S * penalty_S
            G_soft = g_conf * (1.0 - combined_penalty)
            
        elif cfg.soft_aggregation == "min":
            # Minimum (most conservative): use worst-case assessment
            min_survival = torch.min(
                torch.stack([g_conf, 1.0 - penalty_D, 1.0 - penalty_S], dim=0),
                dim=0
            )[0]
            G_soft = min_survival
            
        else:
            raise ValueError(f"Unknown soft_aggregation: {cfg.soft_aggregation}")
        
        G_soft = G_soft.clamp(0.0, 1.0)
        debug["G_soft"] = G_soft
        
        # ========== Stage 3: Final Mask Generation ==========
        # Exclude hard-rejected regions from soft scores
        G_soft_filtered = torch.where(
            hard_reject_mask,
            torch.zeros_like(G_soft),
            G_soft
        )
        debug["G_soft_filtered"] = G_soft_filtered
        
        # Select final mask based on soft scores
        if cfg.soft_selection_mode == "threshold":
            # Simple thresholding
            G_final = (G_soft_filtered >= cfg.soft_threshold).float()
            
        elif cfg.soft_selection_mode == "top_k":
            # Select top-k% pixels by soft score
            G_final = torch.zeros_like(G_soft_filtered, dtype=torch.bool)
            
            for i in range(B):
                soft_flat = G_soft_filtered[i].reshape(-1)
                n_select = int(soft_flat.numel() * cfg.soft_keep_ratio)
                
                if n_select > 0:
                    # Find threshold for top-k%
                    threshold = torch.quantile(
                        soft_flat,
                        1.0 - cfg.soft_keep_ratio,
                        interpolation='higher'
                    )
                    G_final[i] = (G_soft_filtered[i] >= threshold)
            
            G_final = G_final.float()
            
        elif cfg.soft_selection_mode == "adaptive":
            # Adaptive thresholding based on distribution statistics
            G_final = torch.zeros_like(G_soft_filtered, dtype=torch.bool)
            
            for i in range(B):
                soft_flat = G_soft_filtered[i].reshape(-1)
                mean = soft_flat.mean()
                std = soft_flat.std()
                threshold = mean + cfg.adaptive_std_factor * std
                G_final[i] = (G_soft_filtered[i] >= threshold)
            
            G_final = G_final.float()
            
        else:
            raise ValueError(f"Unknown soft_selection_mode: {cfg.soft_selection_mode}")
        
        debug["G_final"] = G_final
        
        # ========== Statistics ==========
        debug["stats"] = {
            "hard_reject_ratio": hard_reject_mask.float().mean().item(),
            "soft_keep_ratio": G_final.mean().item(),
            "high_conf_ratio": high_conf.float().mean().item(),
        }
        
        return (G_final, debug) if return_debug else (G_final, {})


# -------------------- Helper: Visualization --------------------

def visualize_gate_debug(
    debug_dict: Dict[str, Tensor],
    student_input: Tensor,
    student_output: Tensor,
    save_path: str = "gate_debug.png",
    show_stats: bool = True,
):
    """
    Visualize intermediate results from hallucination gate.
    
    Args:
        debug_dict: Debug dictionary from gate.forward(..., return_debug=True)
        student_input: Input noisy image [B,C,H,W]
        student_output: Denoised output [B,C,H,W]
        save_path: Path to save visualization
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Extract first image from batch for visualization
    idx = 0
    
    # Row 1: Inputs and scores
    axes[0, 0].imshow(student_input[idx, 0].cpu().detach(), cmap='gray')
    axes[0, 0].set_title('Student Input (Noisy)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(student_output[idx, 0].cpu().detach(), cmap='gray')
    axes[0, 1].set_title('Student Output (Denoised)')
    axes[0, 1].axis('off')
    
    if 'd_score' in debug_dict:
        im = axes[0, 2].imshow(debug_dict['d_score'][idx, 0].cpu().detach(), cmap='hot')
        axes[0, 2].set_title('D-Gate Score (residual anomaly)')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
    
    # Row 2: S-score, hard reject, high conf
    if 's_score' in debug_dict:
        im = axes[1, 0].imshow(debug_dict['s_score'][idx, 0].cpu().detach(), cmap='hot')
        axes[1, 0].set_title('S-Gate Score (temporal inconsistency)')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    
    axes[1, 1].imshow(debug_dict['hard_reject_total'][idx, 0].cpu().detach(), cmap='Reds')
    axes[1, 1].set_title(f'Hard Reject (worst {debug_dict["stats"]["hard_reject_ratio"]:.1%})')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(debug_dict['high_conf'][idx, 0].cpu().detach(), cmap='Greens')
    axes[1, 2].set_title(f'High Confidence ({debug_dict["stats"]["high_conf_ratio"]:.1%})')
    axes[1, 2].axis('off')
    
    # Row 3: Soft fusion and final
    im = axes[2, 0].imshow(debug_dict['G_soft'][idx, 0].cpu().detach(), cmap='viridis')
    axes[2, 0].set_title('G_soft (before hard reject)')
    axes[2, 0].axis('off')
    plt.colorbar(im, ax=axes[2, 0], fraction=0.046)
    
    im = axes[2, 1].imshow(debug_dict['G_soft_filtered'][idx, 0].cpu().detach(), cmap='viridis')
    axes[2, 1].set_title('G_soft (after hard reject)')
    axes[2, 1].axis('off')
    plt.colorbar(im, ax=axes[2, 1], fraction=0.046)
    
    axes[2, 2].imshow(debug_dict['G_final'][idx, 0].cpu().detach(), cmap='gray')
    axes[2, 2].set_title(f'Final Mask (keep {debug_dict["stats"]["soft_keep_ratio"]:.1%})')
    axes[2, 2].axis('off')
    
    plt.suptitle('Denoising Hallucination Gate - Debug Visualization', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[DONE] Visualization saved to {save_path}")
    
    # save statics to .txt
    stats_path = None
    if show_stats and 'stats' in debug_dict:
        stats_path = save_path.replace('.png', '_stats.txt')
        with open(stats_path, 'w') as f:
            f.write("Gate Statistics\n")
            f.write("="*50 + "\n")
            for key, value in debug_dict['stats'].items():
                f.write(f"{key}: {value}\n")
        print(f"[DONE] Statistics saved to {stats_path}")
    
    return save_path, stats_path
    

def save_gate_comparison(
    iterations: list,
    save_dir: str = "denoise_dncnn_baseline/results",
    filename: str = "gate_comparison.png"
) -> str:
    """
    Create a comparison visualization across multiple RAST iterations.
    
    Args:
        iterations: List of dicts, each containing:
                   - 'iter': iteration number
                   - 'debug': debug dict from gate
                   - 'student_input': input image
                   - 'student_output': output image
        save_dir: Directory to save visualization
        filename: Filename for the comparison
    
    Returns:
        save_path: Full path where the comparison was saved
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    save_path = save_dir_path / filename
    
    n_iters = len(iterations)
    fig, axes = plt.subplots(n_iters, 5, figsize=(20, 4*n_iters))
    
    if n_iters == 1:
        axes = axes.reshape(1, -1)
    
    for i, iter_data in enumerate(iterations):
        iter_num = iter_data['iter']
        debug = iter_data['debug']
        s_input = iter_data['student_input']
        s_output = iter_data['student_output']
        
        # Column 0: Input
        axes[i, 0].imshow(s_input[0, 0].cpu().detach(), cmap='gray')
        axes[i, 0].set_title(f'Iter {iter_num}\nInput', fontweight='bold')
        axes[i, 0].axis('off')
        
        # Column 1: Output
        axes[i, 1].imshow(s_output[0, 0].cpu().detach(), cmap='gray')
        axes[i, 1].set_title(f'Iter {iter_num}\nOutput', fontweight='bold')
        axes[i, 1].axis('off')
        
        # Column 2: D-score
        if 'd_score' in debug:
            im = axes[i, 2].imshow(debug['d_score'][0, 0].cpu(), cmap='hot', vmin=0, vmax=1)
            axes[i, 2].set_title(f'Iter {iter_num}\nD-Score', fontweight='bold')
            plt.colorbar(im, ax=axes[i, 2], fraction=0.046)
        axes[i, 2].axis('off')
        
        # Column 3: S-score
        if 's_score' in debug:
            im = axes[i, 3].imshow(debug['s_score'][0, 0].cpu(), cmap='hot', vmin=0, vmax=1)
            axes[i, 3].set_title(f'Iter {iter_num}\nS-Score', fontweight='bold')
            plt.colorbar(im, ax=axes[i, 3], fraction=0.046)
        axes[i, 3].axis('off')
        
        # Column 4: Final mask
        if 'G_final' in debug:
            axes[i, 4].imshow(debug['G_final'][0, 0].cpu(), cmap='gray')
            keep_ratio = debug.get('stats', {}).get('soft_keep_ratio', 0)
            axes[i, 4].set_title(f'Iter {iter_num}\nMask ({keep_ratio:.1%})', fontweight='bold')
        axes[i, 4].axis('off')
    
    plt.suptitle('RAST Iterations Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[DONE] Comparison saved to {save_path}")
    
    return str(save_path)