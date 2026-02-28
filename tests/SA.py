# SA.py
import torch
class SaturableAbsorber:
    def __init__(self, params: dict[str, float]) -> None:
        self.Tns = params['Tns']
        self.deltaT = params['deltaT']
        self.Esat = params['Esat']
        self.alpha_loss = 0.8

    def apply(self, A: torch.Tensor) -> torch.Tensor:
        I = torch.abs(A)**2
        I_max = torch.max(I)
        if I_max < 1e-12:
            return A
        transmission = torch.sqrt(torch.clamp(
            1 - self.deltaT * torch.exp(-I / I_max) - self.Tns, 
            min=0
        ))

        A = self.alpha_loss * A * transmission  
        self.output = A
        self.intensity = (torch.abs(self.output) ** 2).detach().cpu().numpy() 
        return self.output