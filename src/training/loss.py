import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.4, gamma=2, reduction='mean'):
        """
        Args:
            alpha (float): Weight for the positive class (1). 
                           The weight for class 0 will be (1-alpha).
            gamma (float): Focusing parameter.
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        # Calculate BCE (log(pt))
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        
        # Calculate Alpha Factor: alpha for y=1, (1-alpha) for y=0
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal Loss formula: alpha_t * (1-pt)^gamma * BCE
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss