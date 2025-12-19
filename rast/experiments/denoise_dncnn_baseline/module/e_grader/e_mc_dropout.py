# /workspace/rast/experiments/denoise_dncnn_baseline/module/e_grader/e_mc_dropout.py
import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class MCDropoutEpistemicGrader(nn.Module):
    """
    MC-dropout-based Epistemic Uncertainty Grader
    
    - multiple iterations student forward in reference mode(dropout enabled)
    - adopt runtime injection(add dropout layer after ReLU)
    - calculate the variance of dropout samples as epistemic uncertainty
    
    workflow:
    1. input the input image of this iteration(x_prev)
    2. T times runtime MC Dropout forward sampling
    3. calculate the variance of these T samples
    4. return pixel-wise epistemic uncertainty map
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        dropout_rate: float = 0.1,
        n_samples: int = 10,
        target_layer_types: tuple = (nn.ReLU,)
    ):
        """
        Args:
            student_model: pre-trained student model(the same model plugged in the STUDENT module)
            dropout_rate: 0.05-0.2 in default
            n_samples: sampling times, 5-20 in default
            target_layer_types: inject dropout layer after which layer(ReLU in default)
        """
        super().__init__()
        self.student_model = student_model
        self.dropout_rate = dropout_rate
        self.n_samples = n_samples
        self.target_layer_types = target_layer_types
        
        # saved hooks dict, for cleanup
        self.hooks = []
        
    def _dropout_hook(self, module, input, output):
        """
        Hook func, the key config of dropout application
        """
        return nn.functional.dropout2d(
            output,
            p=self.dropout_rate,
            training=True,  # force the use of dropout even in eval
            inplace=False
        )
    
    def _register_dropout_hooks(self):
        """
        inject dropout hooks after certain layers in student model
        """
        self.hooks = []
        for name, module in self.student_model.named_modules():
            if isinstance(module, self.target_layer_types):
                hook = module.register_forward_hook(self._dropout_hook)
                self.hooks.append(hook)
        
        if len(self.hooks) == 0:
            print(f"[ERROR] Layer {self.target_layer_types} NOT FOUND")
            print("MC Dropout failed!")
    
    def _remove_dropout_hooks(self):
        """
        cleanup all injected hooks
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def forward(
        self,
        s_img_output: torch.Tensor,
        *,
        student_input: Optional[torch.Tensor] = None,
        features: Optional[Dict[str, torch.Tensor]] = None,
        aux: Any = None
    ) -> torch.Tensor:
        """
        calculate epistemic uncertainty
        
        Args:
            s_img_output: not used(interface reserved)
            features: not used(interface reserved)
            aux: 
                - x_prev: input noisy image of this t iteration
                - X0: ground truth(if available)
                - mask_evidence: reserved areas
        
        Returns:
            epistemic_map: [B, 1, H, W] epistemic uncertainty map
        """
        # input image of current iteration
        x_current = student_input  # [B, C, H, W]
        # value guard
        if x_current is None:
            raise ValueError("student_input is required for MC Dropout grader")
        
        # Step 1: dropout hooks injection
        self._register_dropout_hooks()
        
        # Step 2: MC Dropout sampling
        samples = []
        
        # turn on eval mode(dropout enforced)
        original_training = self.student_model.training
        self.student_model.eval()
        
        try:
            with torch.no_grad():
                for i in range(self.n_samples):
                    # save the output(samples) after every forward
                    pred = self.student_model(x_current)
                    samples.append(pred)
            
            # Step 3: calculate variance
            # samples: List of [B, C, H, W]
            samples_tensor = torch.stack(samples, dim=0)  # [T, B, C, H, W]
            
            # calculate the pixel-wise uncertainty
            variance = torch.var(samples_tensor, dim=0)  # [B, C, H, W]
            
            # if multi-channel，average into single channel
            if variance.size(1) > 1:
                epistemic_map = variance.mean(dim=1, keepdim=True)  # [B, 1, H, W]
            else:
                epistemic_map = variance
            
            # Step 4: normalize into [0, 1]
            # robust normalization
            eps = 1e-8
            p5 = torch.quantile(epistemic_map, 0.05)
            p95 = torch.quantile(epistemic_map, 0.95)
            epistemic_map = (epistemic_map - p5) / (p95 - p5 + eps)
            epistemic_map = torch.clamp(epistemic_map, 0, 1)
            
        finally:
            # Step 5: hooks cleanup & model config recovery
            self._remove_dropout_hooks()
            self.student_model.train(original_training)
        
        return epistemic_map
    
    def get_mc_predictions(
        self,
        x: torch.Tensor,
        return_mean: bool = False
    ) -> torch.Tensor:
        """
        helpers: get all predictions in mc dropout mode
        
        Args:
            x: input image
            return_mean: Whether to return the average prediction (optional to replace the deterministic output of student)
        
        Returns:
            return_mean=False: all dropout samples [T, B, C, H, W]
            return_mean=True: prediction in average [B, C, H, W]
        """
        self._register_dropout_hooks()
        original_training = self.student_model.training
        self.student_model.eval()
        
        try:
            with torch.no_grad():
                samples = []
                for _ in range(self.n_samples):
                    pred = self.student_model(x)
                    samples.append(pred)
                
                samples_tensor = torch.stack(samples, dim=0)
                
                if return_mean:
                    return samples_tensor.mean(dim=0)
                else:
                    return samples_tensor
        finally:
            self._remove_dropout_hooks()
            self.student_model.train(original_training)


class LightweightMCDropoutGrader(MCDropoutEpistemicGrader):
    """
    Lightweight version: Reduce the number of samples to increase speed(for baseline quick test)
    """
    def __init__(
        self,
        student_model: nn.Module,
        dropout_rate: float = 0.1,
        n_samples: int = 3  # only sampling in 3 times
    ):
        super().__init__(
            student_model=student_model,
            dropout_rate=dropout_rate,
            n_samples=n_samples
        )

# import only
if __name__ == "__main__":
    print(f"======================= MC Dropout epistemic uncertainty grader =======================")
    print(f"Example:")
    print(f"[e-grader construction]")
    print(f"e_grader = MCDropoutEpistemicGrader(")
    print(f"    student_model=student,")
    print(f"    dropout_rate=0.1,")
    print(f"    n_samples=10 (=3 for lightweight version)")
    print(f")")
    print()
    print(f"[e-grader usage]")
    print(f"with torch.no_grad():")
    print(f"    epistemic_map = e_grader(")
    print(f"    s_img_output=None,")
    print(f"    features=None,")
    print(f"    aux=aux")
    print(f")")
    print(f"=======================================================================================")