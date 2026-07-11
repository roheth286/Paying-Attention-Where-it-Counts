import torch

class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.loss_sum = 0
        self.count = 0

    def update(self, tp, fp, fn, loss, batch_size):
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.loss_sum += loss * batch_size
        self.count += batch_size

    def compute(self):
        eps = 1e-7
        dice = (2 * self.tp) / (2 * self.tp + self.fp + self.fn + eps)
        loss = self.loss_sum / self.count
        return dice, loss


def compute_batch_statistics(logits, targets, ignore_index=255, threshold=0.5):
    # Remove channel dim
    logits = logits.squeeze(1)

    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(logits)

    # Binary prediction
    preds = (probs > threshold).long()

    # Mask out ignore pixels
    valid_mask = (targets != ignore_index)

    preds = preds[valid_mask]
    targets = targets[valid_mask]

    # Compute statistics
    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    return tp, fp, fn
