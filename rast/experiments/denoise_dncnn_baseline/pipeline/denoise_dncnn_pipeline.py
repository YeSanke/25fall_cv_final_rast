"""
DnCNN-based Denoising RAST Pipeline
=====================================

Complete pipeline assembly for image denoising task using DnCNN as student backbone.

Components:
- Student: DnCNN backbone with pre/post normalization
- E-Grader: MC Dropout for epistemic uncertainty
- A-Grader: VAE for aleatoric uncertainty  
- H-Gate: Denoising hallucination gate (D-Gate + S-Gate)
- Assembler: Standard image composition logic
- Engine: RAST loop orchestrator

Usage:
    # Quick start from pretrained checkpoints
    pipeline = DenoisingPipeline.from_pretrained(
        dncnn_path='checkpoints/dncnn_sigma25.pth',
        vae_path='checkpoints/vae_denoising.pth',
        steps=5
    )
    
    # Denoise a noisy image
    noisy = torch.randn(1, 1, 256, 256) * 0.1  # Example noisy image
    denoised = pipeline.denoise(noisy, noise_sigma=25.0)
    
    # Access detailed results
    denoised, traces = pipeline.denoise(noisy, noise_sigma=25.0, return_traces=True)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

# ===== Setup import paths =====
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ===== Core RAST components (from project root) =====
from core.engine import RASTEngine
from core.aux import Aux
from core.assembler import DenoisingAssembler
from core.student import Student, Normalizer
from core.student_pipeline import StudentPipeline
from core.teacher import Teacher

# ===== Task-specific components (from experiment folder) =====
# Student backbone
from experiments.denoise_dncnn_baseline.networks.dncnn.model import DnCNN

# Uncertainty graders
from experiments.denoise_dncnn_baseline.networks.vae.model import VAE
from experiments.denoise_dncnn_baseline.module.e_grader.e_mc_dropout import (
    MCDropoutEpistemicGrader, 
    LightweightMCDropoutGrader
)

# Hallucination gate
from experiments.denoise_dncnn_baseline.module.h_gate.denoise_hallucination_gate import (
    DenoiseHallucinationGate,
    DenoiseGateConfig,
)


# ======================================================================
#                       VAE-based A-Grader Wrapper
# ======================================================================

class VAEAleatoricGrader(nn.Module):
    """
    VAE-based Aleatoric Uncertainty Grader.
    
    Wraps the VAE model to conform to the UncertaintyGrader interface.
    Uses reconstruction variance as aleatoric uncertainty estimate.
    """
    
    def __init__(self, vae_model: VAE):
        """
        Args:
            vae_model: Pre-trained VAE model
        """
        super().__init__()
        self.vae = vae_model
        
    def forward(
        self,
        student_output: Tensor,
        *,
        student_input: Optional[Tensor] = None,
        features: Optional[Dict[str, Tensor]] = None,
        aux: Optional[Any] = None,
    ) -> Tensor:
        """
        Compute aleatoric uncertainty from VAE reconstruction variance.
        
        Args:
            student_output: Student's denoised output [B,C,H,W]
            student_input: (Optional) Not used for VAE grader
            features: (Optional) Not used for VAE grader
            aux: (Optional) Auxiliary context
            
        Returns:
            Aleatoric uncertainty map [B,1,H,W] in [0,1]
        """
        # VAE forward: get reconstruction mean and log variance
        with torch.no_grad():
            recon_mean, recon_logvar, mu, logvar = self.vae(student_output)
        
        # Convert log variance to variance
        recon_var = torch.exp(recon_logvar)  # [B, C, H, W]
        
        # Aggregate across channels (if multi-channel)
        if recon_var.size(1) > 1:
            aleatoric_map = recon_var.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        else:
            aleatoric_map = recon_var
        
        # Normalize to [0, 1] using robust percentile normalization
        eps = 1e-8
        p5 = torch.quantile(aleatoric_map, 0.05)
        p95 = torch.quantile(aleatoric_map, 0.95)
        aleatoric_map = (aleatoric_map - p5) / (p95 - p5 + eps)
        aleatoric_map = torch.clamp(aleatoric_map, 0, 1)
        
        return aleatoric_map


# ======================================================================
#                       Pipeline Assembly Functions
# ======================================================================

def build_student_pipeline(
    dncnn_backbone: DnCNN,
    *,
    amp: bool = False,
    pad_divisible: int = 16,
) -> StudentPipeline:
    """
    Build StudentPipeline with DnCNN backbone.
    
    Pipeline structure:
        input → [normalize] → [pad] → DnCNN → [denormalize] → [crop] → output
    
    Args:
        dncnn_backbone: Pre-trained DnCNN model
        amp: Whether to use automatic mixed precision
        pad_divisible: Pad input to be divisible by this value
        
    Returns:
        StudentPipeline ready for RAST loop
    """
    
    # Wrap DnCNN backbone in Student module
    student = Student(
        backbone=dncnn_backbone,
        amp=amp,
        feature_fn=None,  # DnCNN doesn't need intermediate features
        register_hooks=False,
    )
    
    # Assemble complete pipeline with pre/post processing
    pipeline = StudentPipeline(
        backbone=student,
        preproc=None,
        postproc=None,
        pad_divisible=pad_divisible,
        amp=amp,
    )
    
    return pipeline


def build_teacher(
    vae_model: VAE,
    dncnn_backbone: DnCNN,
    *,
    gate_config: Optional[DenoiseGateConfig] = None,
    use_lightweight_mc: bool = False,
    mc_dropout_rate: float = 0.1,
    mc_n_samples: int = 10,
    enable_graders: bool = True,
) -> Teacher:
    """
    Build Teacher with dual uncertainty graders and hallucination gate.
    
    Teacher architecture:
        student_output → [E-Grader: MC Dropout] → u_e
                      → [A-Grader: VAE]        → u_a
                      → [Combine]              → c_map
                      → [H-Gate: D+S Gate]     → G_t
    
    Args:
        vae_model: Pre-trained VAE for aleatoric grading
        dncnn_backbone: DnCNN model for MC dropout epistemic grading
        gate_config: Configuration for hallucination gate
        use_lightweight_mc: Use lightweight MC dropout (3 samples) for faster inference
        mc_dropout_rate: Dropout rate for MC sampling
        mc_n_samples: Number of MC samples (ignored if use_lightweight_mc=True)
        enable_graders: If False, skip epistemic/aleatoric graders (full trust mode)
        
    Returns:
        Teacher with configured graders and gate
    """
    # ===== Build Aleatoric Grader (VAE) =====
    a_grader = VAEAleatoricGrader(vae_model=vae_model)
    
    # ===== Build Epistemic Grader (MC Dropout) =====
    if use_lightweight_mc:
        e_grader = LightweightMCDropoutGrader(
            student_model=dncnn_backbone,
            dropout_rate=mc_dropout_rate,
            n_samples=3,  # Fixed to 3 for lightweight version
        )
    else:
        e_grader = MCDropoutEpistemicGrader(
            student_model=dncnn_backbone,
            dropout_rate=mc_dropout_rate,
            n_samples=mc_n_samples,
        )
    
    # ===== Build Graders (if enable_graders is True) =====
    if enable_graders: # enable_graders is True
        # Aleatoric Grader (VAE)
        a_grader = VAEAleatoricGrader(vae_model=vae_model)
        
        # Epistemic Grader (MC Dropout)
        if use_lightweight_mc:
            e_grader = LightweightMCDropoutGrader(
                student_model=dncnn_backbone,
                dropout_rate=mc_dropout_rate,
                n_samples=3,
            )
        else:
            e_grader = MCDropoutEpistemicGrader(
                student_model=dncnn_backbone,
                dropout_rate=mc_dropout_rate,
                n_samples=mc_n_samples,
            )
    else:
        # enable_graders is False
        e_grader = None
        a_grader = None
    
    # ===== Build Hallucination Gate (If ate_config.enabled is True) =====
    if gate_config is None or not gate_config.enabled:
        h_gate = None
    else:
        h_gate = DenoiseHallucinationGate(config=gate_config)
    
    # ===== Assemble Teacher =====
    teacher = Teacher(
        e_grader=e_grader,
        a_grader=a_grader,
        hallucination_gate=h_gate,
        combine_fn=None,
    )
    
    return teacher


def build_rast_engine(
    student_pipeline: StudentPipeline,
    teacher: Teacher,
    *,
    steps: int = 5,
) -> RASTEngine:
    """
    Build complete RAST engine.
    
    Args:
        student_pipeline: Prepared student pipeline
        teacher: Configured teacher with graders and gate
        steps: Number of RAST iterations
        
    Returns:
        RASTEngine ready for inference
    """
    assembler = DenoisingAssembler()
    
    engine = RASTEngine(
        student=student_pipeline,
        teacher=teacher,
        assembler=assembler,
        steps=steps,
    )
    
    return engine


# ======================================================================
#                       High-Level Pipeline Class
# ======================================================================

class DenoisingPipeline:
    """
    High-level wrapper for DnCNN-based denoising RAST pipeline.
    
    Provides convenient interfaces for:
    - Loading from pretrained checkpoints
    - Creating from existing components
    - Running inference with automatic aux management
    - Extracting intermediate traces for analysis
    
    Example:
        # Create from checkpoints
        pipeline = DenoisingPipeline.from_pretrained(
            dncnn_path='checkpoints/dncnn.pth',
            vae_path='checkpoints/vae.pth'
        )
        
        # Denoise an image
        denoised = pipeline.denoise(noisy_image, noise_sigma=25.0)
        
        # Get detailed traces
        result, traces = pipeline.denoise(
            noisy_image, 
            noise_sigma=25.0,
            return_traces=True
        )
    """
    
    def __init__(
        self,
        engine: RASTEngine,
        *,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Initialize pipeline with a pre-built RASTEngine.
        
        Args:
            engine: Configured RASTEngine
            device: Device to run inference on
        """
        self.engine = engine
        self.device = device
        self.engine.to(device)
        self.engine.eval()
        
    
    @classmethod
    def from_pretrained(
        cls,
        dncnn_path: str,
        vae_path: str,
        *,
        steps: int = 5,
        gate_config: Optional[DenoiseGateConfig] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_lightweight_mc: bool = False,
        amp: bool = False,
        enable_graders: bool = True,
    ) -> DenoisingPipeline:
        """
        Create pipeline from pretrained checkpoint paths.
        
        Args:
            dncnn_path: Path to DnCNN checkpoint (.pth file)
            vae_path: Path to VAE checkpoint (.pth file)
            steps: Number of RAST iterations
            gate_config: Custom gate configuration (uses defaults if None)
            device: Device to load models on
            use_lightweight_mc: Use fast 3-sample MC dropout
            amp: Enable automatic mixed precision
            enable_graders: If False, skip uncertainty graders (full trust mode)
            
        Returns:
            Initialized DenoisingPipeline
            
        Raises:
            FileNotFoundError: If checkpoint files don't exist
        """
        # ===== Validate paths =====
        dncnn_path = Path(str(dncnn_path))
        vae_path = Path(str(vae_path))
        
        if not dncnn_path.exists():
            raise FileNotFoundError(f"DnCNN checkpoint not found: {dncnn_path}")
        if not vae_path.exists():
            raise FileNotFoundError(f"VAE checkpoint not found: {vae_path}")
        
        # ===== Load DnCNN =====
        print(f"[INFO] Loading DnCNN from {dncnn_path}")
        dncnn = DnCNN(channels=1, num_layers=17)  # Standard DnCNN-17 for grayscale
        dncnn.load_state_dict(torch.load(str(dncnn_path), map_location=device))
        dncnn.eval()
        
        # ===== Load VAE =====
        print(f"[INFO] Loading VAE from {vae_path}")
        vae = VAE(in_channels=1, latent_dim=16, base_channels=32)
        vae.load_state_dict(torch.load(str(vae_path), map_location=device))
        vae.eval()
        
        # ===== Build components =====
        print("[INFO] Building student pipeline...")
        student_pipeline = build_student_pipeline(
            dncnn_backbone=dncnn,
            amp=amp,
            pad_divisible=16,
        )
        
        print("[INFO] Building teacher...")
        teacher = build_teacher(
            vae_model=vae,
            dncnn_backbone=dncnn,
            gate_config=gate_config,
            use_lightweight_mc=use_lightweight_mc,
            enable_graders=enable_graders,
        )
        
        print(f"[INFO] Building RAST engine ({steps} steps)...")
        engine = build_rast_engine(
            student_pipeline=student_pipeline,
            teacher=teacher,
            steps=steps,
        )
        
        print(f"[DONE] Pipeline ready on {device}")
        return cls(engine=engine, device=device)
    
    @classmethod
    def from_components(
        cls,
        dncnn_backbone: DnCNN,
        vae_model: VAE,
        *,
        steps: int = 5,
        gate_config: Optional[DenoiseGateConfig] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_lightweight_mc: bool = False,
        amp: bool = False,
    ) -> DenoisingPipeline:
        """
        Create pipeline from already-loaded model components.
        
        Args:
            dncnn_backbone: Loaded DnCNN model
            vae_model: Loaded VAE model
            steps: Number of RAST iterations
            gate_config: Custom gate configuration
            device: Device to run on
            use_lightweight_mc: Use fast MC dropout
            amp: Enable AMP
            
        Returns:
            Initialized DenoisingPipeline
        """
        print("[INFO] Building from provided components...")
        
        # Move models to device
        dncnn_backbone = dncnn_backbone.to(device).eval()
        vae_model = vae_model.to(device).eval()
        
        # Build pipeline
        student_pipeline = build_student_pipeline(
            dncnn_backbone=dncnn_backbone,
            amp=amp,
        )
        
        teacher = build_teacher(
            vae_model=vae_model,
            dncnn_backbone=dncnn_backbone,
            gate_config=gate_config,
            use_lightweight_mc=use_lightweight_mc,
        )
        
        engine = build_rast_engine(
            student_pipeline=student_pipeline,
            teacher=teacher,
            steps=steps,
        )
        
        print(f"[DONE] Pipeline ready on {device}")
        return cls(engine=engine, device=device)
    
    def create_aux(
        self,
        noisy_image: Tensor,
        *,
        noise_sigma: float = 25.0,
        mask: Optional[Tensor] = None,
        confidence_threshold: float = 0.5,
        max_history_length: int = 10,
    ) -> Aux:
        """
        Create auxiliary context for RAST loop.
        
        Args:
            noisy_image: Noisy input image [B,C,H,W]
            noise_sigma: Noise level (for metadata)
            mask: Optional observation mask [B,1,H,W] (all zeros = no evidence)
            confidence_threshold: Threshold for high-confidence detection
            max_history_length: Maximum S-gate history length
            
        Returns:
            Aux object ready for engine.run()
        """
        B, C, H, W = noisy_image.shape
        
        # Default mask: no evidence (all regions are uncertain)
        if mask is None:
            mask = torch.zeros(B, 1, H, W, device=noisy_image.device)
        
        # Create aux with proper initialization
        aux = Aux(
            X0=noisy_image.clone(),  # Original observation
            M0=torch.zeros(B, 1, H, W, device=noisy_image.device),  # Evidence mask
            student_input=noisy_image.clone(),  # Initial input to student
            x_prev=None,  # Will be set after first iteration
            pipe_meta={'noise_sigma': noise_sigma},  # Metadata
            teacher_output_mask=None,  # Will be set by teacher
            history_high_conf_masks=[],  # Legacy field
            historical_images=[],  # For S-gate
            historical_confidence_maps=[],  # For S-gate
            max_history_length=max_history_length,
            confidence_threshold=confidence_threshold,
        )
        
        return aux
    
    @torch.no_grad()
    def denoise(
        self,
        noisy_image: Tensor,
        *,
        noise_sigma: float = 25.0,
        mask: Optional[Tensor] = None,
        return_traces: bool = False,
        return_gate_debug: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Any]]:
        """
        Denoise an image using RAST pipeline.
        
        Args:
            noisy_image: Noisy input image [B,C,H,W] or [C,H,W] or [H,W]
            noise_sigma: Noise level (for reference, not used in inference)
            mask: Optional evidence mask [B,1,H,W]
            return_traces: If True, return intermediate results
            return_gate_debug: If True, collect gate debug info for each iteration
            
        Returns:
            If return_traces=False and return_gate_debug=False:
                denoised: Final denoised image [B,C,H,W]
            Otherwise:
                (denoised, results): Final output + dict containing:
                    - traces: Dict of intermediate results (if return_traces=True)
                    - gate_debug_data: List of gate debug dicts (if return_gate_debug=True)
        """
        # Ensure 4D tensor [B,C,H,W]
        if noisy_image.dim() == 2:  # [H,W]
            noisy_image = noisy_image.unsqueeze(0).unsqueeze(0)
        elif noisy_image.dim() == 3:  # [C,H,W]
            noisy_image = noisy_image.unsqueeze(0)
        
        # Move to device
        noisy_image = noisy_image.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        
        # Create aux
        aux = self.create_aux(
            noisy_image=noisy_image,
            noise_sigma=noise_sigma,
            mask=mask,
        )
        
        # Store gate debug data if requested
        gate_debug_data = [] if return_gate_debug else None
        
        # Monkey-patch teacher to collect gate debug info
        if return_gate_debug:
            original_grading = self.engine.teacher.grading
            
            def grading_with_debug(student_output, aux, **kwargs):
                # Call original grading with debug enabled
                c_map, G_t = original_grading(student_output, aux, **kwargs)
                
                # Collect debug info from hallucination gate
                if hasattr(self.engine.teacher, 'hallucination_gate'):
                    _, debug = self.engine.teacher.hallucination_gate(
                        student_input=aux.student_input,
                        student_output=student_output,
                        c_map=c_map,
                        aux=aux,
                        return_debug=True,
                    )
                    
                    gate_debug_data.append({
                        'iter': len(gate_debug_data) + 1,
                        'debug': debug,
                        'student_input': aux.student_input.detach().cpu(),
                        'student_output': student_output.detach().cpu(),
                    })
                
                return c_map, G_t
            
            # Temporarily replace grading method
            self.engine.teacher.grading = grading_with_debug
        
        try:
            # Run RAST loop
            denoised, traces = self.engine.run(aux, return_traces=True)
        finally:
            # Restore original grading method
            if return_gate_debug:
                self.engine.teacher.grading = original_grading
        
        # Prepare return values
        if not return_traces and not return_gate_debug:
            return denoised
        
        results = {}
        if return_traces:
            results['traces'] = traces
        if return_gate_debug:
            results['gate_debug_data'] = gate_debug_data
        
        return denoised, results
    
    @torch.no_grad()
    def denoise_and_save(
        self,
        noisy_image: Tensor,
        *,
        noise_sigma: float = 25.0,
        mask: Optional[Tensor] = None,
        save_dir: str = "denoise_dncnn_baseline/results",
        experiment_name: Optional[str] = None,
        save_traces: bool = True,
        save_gate_debug: bool = True,
        metrics: Optional[Dict[str, float]] = None,
        notes: Optional[str] = None,
    ) -> Tuple[Tensor, Path]:
        """
        Denoise an image and automatically save all results.
        
        This is a convenience method that combines denoising with result saving.
        
        Args:
            noisy_image: Noisy input image
            noise_sigma: Noise level
            mask: Optional evidence mask
            save_dir: Base directory for saving results
            experiment_name: Optional experiment name (default: timestamp)
            save_traces: If True, save RAST traces
            save_gate_debug: If True, save gate debug visualizations
            metrics: Optional metrics to save (e.g., {"psnr": 28.5})
            notes: Optional experiment notes
            
        Returns:
            (denoised, run_dir): Final output and path to run directory
            
        Example:
            denoised, run_dir = pipeline.denoise_and_save(
                noisy_image,
                noise_sigma=25.0,
                metrics={"psnr": 28.5, "ssim": 0.85},
                notes="Testing new gate config"
            )
        """
        from rast.utils.result_manager import ResultManager
        
        # Create result manager
        manager = ResultManager(
            base_dir=save_dir,
            experiment_name=experiment_name,
            auto_create=True,
        )
        
        # Denoise with full debug info
        denoised, results = self.denoise(
            noisy_image,
            noise_sigma=noise_sigma,
            mask=mask,
            return_traces=save_traces,
            return_gate_debug=save_gate_debug,
        )
        
        # Save input and output images
        manager.save_image(noisy_image, "input.png")
        manager.save_image(denoised, "output.png")
        
        # Save intermediate outputs
        if save_traces and 'traces' in results:
            traces = results['traces']
            for i, s_output in enumerate(traces.get("student_outputs", [])):
                manager.save_image(s_output, f"iter_{i+1}_output.png")
            
            # Save full traces
            manager.save_traces(traces)
        
        # Save gate visualizations
        if save_gate_debug and 'gate_debug_data' in results:
            gate_data = results['gate_debug_data']
            
            # Save individual iteration debug plots
            for data in gate_data:
                manager.save_gate_debug(
                    debug_dict=data['debug'],
                    student_input=data['student_input'],
                    student_output=data['student_output'],
                    iteration=data['iter'],
                )
            
            # Save comparison plot
            if len(gate_data) > 1:
                manager.save_gate_comparison(gate_data)
        
        # Save configuration
        config = self.get_config()
        config['noise_sigma'] = noise_sigma
        manager.save_config(config)
        
        # Save metrics
        if metrics:
            manager.save_metrics(metrics)
        
        # Finalize
        manager.finalize(config=config, metrics=metrics, notes=notes)
        
        return denoised, manager.run_dir
    
    def get_config(self) -> Dict[str, Any]:
        """
        Export pipeline configuration as a dictionary.
        
        Returns:
            Dict containing pipeline configuration
        """
        config = {
            'pipeline': 'DenoisingPipeline',
            'device': str(self.device),
            'rast_steps': self.engine.steps,
        }
        
        # Add gate configuration if available
        if hasattr(self.engine.teacher, 'hallucination_gate'):
            gate = self.engine.teacher.hallucination_gate
            if hasattr(gate, 'config'):
                gate_cfg = gate.config
                config['gate'] = {
                    'type': 'DenoiseHallucinationGate',
                    'enabled': list(gate_cfg.enabled) if gate_cfg.enabled else [],
                    'hard_reject_ratio': gate_cfg.hard_reject_ratio,
                    'soft_keep_ratio': gate_cfg.soft_keep_ratio,
                    'confidence_threshold': gate_cfg.confidence_threshold,
                    'soft_aggregation': gate_cfg.soft_aggregation,
                    's_gate_warmup_iterations': gate_cfg.s_gate_warmup_iterations,
                    's_gate_enable_iterations': gate_cfg.s_gate_enable_iterations,
                }
        
        # Add grader information
        config['graders'] = {}
        if self.engine.teacher.e_grader is not None:
            e_grader = self.engine.teacher.e_grader
            config['graders']['epistemic'] = {
                'type': e_grader.__class__.__name__,
                'n_samples': getattr(e_grader, 'n_samples', None),
                'dropout_rate': getattr(e_grader, 'dropout_rate', None),
            }
        
        if self.engine.teacher.a_grader is not None:
            a_grader = self.engine.teacher.a_grader
            config['graders']['aleatoric'] = {
                'type': a_grader.__class__.__name__,
            }
        
        return config
    
    def __call__(self, *args, **kwargs):
        """Alias for denoise() method."""
        return self.denoise(*args, **kwargs)


# ======================================================================
#                       Demo & Testing
# ======================================================================

def demo_usage():
    """
    Demonstrate pipeline usage with synthetic data.
    """
    print("="*70)
    print(" DnCNN Denoising RAST Pipeline - Demo")
    print("="*70)
    
    # ===== Setup =====
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[INFO] Using device: {device}")
    
    # ===== Create synthetic noisy image =====
    # In practice, replace this with real noisy images
    torch.manual_seed(42)
    clean_image = torch.zeros(1, 1, 128, 128)  # Simple test pattern
    clean_image[:, :, 40:88, 40:88] = 1.0  # White square
    
    noise_sigma = 25.0 / 255.0  # Normalize to [0,1] range
    noise = torch.randn_like(clean_image) * noise_sigma
    noisy_image = (clean_image + noise).clamp(0, 1)
    
    print(f"\n[INFO] Created synthetic noisy image: {noisy_image.shape}")
    print(f"       Noise sigma: {noise_sigma:.4f}")
    
    # ===== Option 1: From pretrained (requires checkpoint files) =====
    """
    pipeline = DenoisingPipeline.from_pretrained(
        dncnn_path='checkpoints/dncnn_sigma25.pth',
        vae_path='checkpoints/vae_denoising.pth',
        steps=5,
        device=device,
        use_lightweight_mc=True,  # Fast inference
    )
    """
    
    # ===== Option 2: From components (demo with random init) =====
    print("\n[INFO] Building pipeline from randomly initialized models (demo only)...")
    
    dncnn = DnCNN(channels=1, num_layers=17)
    vae = VAE(in_channels=1, latent_dim=16, base_channels=32)
    
    pipeline = DenoisingPipeline.from_components(
        dncnn_backbone=dncnn,
        vae_model=vae,
        steps=5,
        device=device,
        use_lightweight_mc=True,
    )
    
    # ===== Demo 1: Basic inference =====
    print("\n[INFO] Demo 1: Basic inference...")
    
    denoised = pipeline.denoise(noisy_image, noise_sigma=noise_sigma * 255)
    print(f"       Output shape: {denoised.shape}")
    
    # ===== Demo 2: Inference with traces =====
    print("\n[INFO] Demo 2: Inference with traces...")
    
    denoised, results = pipeline.denoise(
        noisy_image,
        noise_sigma=noise_sigma * 255,
        return_traces=True,
    )
    
    traces = results['traces']
    print(f"       Iterations: {len(traces['student_outputs'])}")
    
    # Show trace statistics
    print("\n[INFO] Trace statistics:")
    for i, (s_out, c_map, g_mask) in enumerate(zip(
        traces['student_outputs'],
        traces['confidence_maps'],
        traces['gate_masks']
    )):
        conf_mean = c_map.mean().item()
        gate_ratio = g_mask.mean().item()
        print(f"  Iter {i+1}: Avg confidence={conf_mean:.3f}, Gate pass ratio={gate_ratio:.3f}")
    
    # ===== Demo 3: Inference with gate debug =====
    print("\n[INFO] Demo 3: Inference with gate debug...")
    
    denoised, results = pipeline.denoise(
        noisy_image,
        noise_sigma=noise_sigma * 255,
        return_traces=True,
        return_gate_debug=True,
    )
    
    gate_debug_data = results['gate_debug_data']
    print(f"       Collected {len(gate_debug_data)} gate debug records")
    
    # ===== Demo 4: Denoise and auto-save =====
    print("\n[INFO] Demo 4: Denoise and auto-save results...")
    
    # Calculate a fake metric for demo
    mse = torch.mean((denoised - clean_image) ** 2).item()
    psnr = 10 * torch.log10(1.0 / (mse + 1e-10)).item()
    
    denoised, run_dir = pipeline.denoise_and_save(
        noisy_image,
        noise_sigma=noise_sigma * 255,
        save_dir="denoise_dncnn_baseline/results",
        experiment_name="demo_run",
        save_traces=True,
        save_gate_debug=True,
        metrics={"psnr": psnr, "mse": mse},
        notes="Demo run with synthetic data",
    )
    
    print(f"       Results saved to: {run_dir}")
    
    # ===== Demo 5: Export configuration =====
    print("\n[INFO] Demo 5: Export configuration...")
    
    config = pipeline.get_config()
    print(f"       Pipeline config keys: {list(config.keys())}")
    print(f"       RAST steps: {config['rast_steps']}")
    if 'gate' in config:
        print(f"       Gate enabled: {config['gate']['enabled']}")
    
    print("\n" + "="*70)
    print(" All demos completed successfully!")
    print("="*70)


if __name__ == "__main__":
    demo_usage()