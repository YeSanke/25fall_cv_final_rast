## RAST: Recurrent Assembler–Student–Teacher Framework
Hexiao Huang hh3464

RAST is a modular, uncertainty-aware iterative framework for image reconstruction and restoration.
It aims to improve reconstruction quality while explicitly controlling hallucinations through
confidence-guided feedback.

The framework was initially developed for MRI reconstruction and is designed to generalize to
other image-to-image tasks such as denoising, super-resolution, and inpainting.


# Overview

RAST decomposes iterative reconstruction into four loosely coupled components:

- **Student**  
  A task-agnostic image-to-image model (e.g., UNet, Transformer-based models) responsible for
  generating reconstructions.

- **Teacher**  
  An uncertainty estimation module that produces pixel-wise confidence maps, capturing both
  aleatoric uncertainty (data noise) and epistemic uncertainty (data incompleteness).

- **Assembler**  
  A confidence-aware aggregation module that selectively preserves reliable predictions,
  enforces data consistency, and suppresses high-confidence hallucinations.

- **Recurrent Loop**  
  The reconstruction is refined over multiple iterations using Teacher-guided feedback rather
  than blind self-refinement.

The design emphasizes modularity, interpretability, and robustness under distribution shift.
