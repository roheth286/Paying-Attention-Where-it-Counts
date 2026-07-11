import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    def __init__(self, ignore_index=255, smooth=1e-7):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        logits = logits.squeeze(1)  # (B, H, W)

        # Create valid mask
        valid_mask = (targets != self.ignore_index)

        # Filter valid pixels
        logits = logits[valid_mask]
        targets = targets[valid_mask].float()

        # ----- BCE -----
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction='mean'
        )

        # ----- Dice -----
        probs = torch.sigmoid(logits)

        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice

        return bce + dice_loss
