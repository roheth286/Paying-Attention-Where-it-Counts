import time
import random
import numpy as np
import torch
import torch.optim as optim
from .metrics import MetricTracker, compute_batch_statistics
from .loss import BCEDiceLoss

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    tracker = MetricTracker()

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = loss_fn(logits, masks)

        loss.backward()
        optimizer.step()

        tp, fp, fn = compute_batch_statistics(logits, masks)
        tracker.update(tp, fp, fn, loss.item(), images.size(0))

    return tracker.compute()


def validate_one_epoch(model, loader, loss_fn, device):
    model.eval()
    tracker = MetricTracker()

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = loss_fn(logits, masks)

            tp, fp, fn = compute_batch_statistics(logits, masks)
            tracker.update(tp, fp, fn, loss.item(), images.size(0))

    return tracker.compute()


def train_model(seed, model_class, train_loader, val_loader, device, epochs=20):
    print(f"\n========== Seed {seed} ==========")
    set_seed(seed)

    model = model_class().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    loss_fn = BCEDiceLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": [],
        "epoch_time": []
    }

    best_val_dice = 0
    best_epoch = 0

    for epoch in range(epochs):
        start_time = time.time()

        train_dice, train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )

        val_dice, val_loss = validate_one_epoch(
            model, val_loader, loss_fn, device
        )

        scheduler.step(val_dice)

        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_time = time.time() - start_time

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)
        history["epoch_time"].append(epoch_time)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            # Create outputs/models folder if not existing
            import os
            os.makedirs(os.path.join("outputs", "models"), exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join("outputs", "models", f"best_model_seed_{seed}.pt")
            )

        print(f"Epoch {epoch+1:02d} | "
              f"Train Dice: {train_dice:.4f} | "
              f"Val Dice: {val_dice:.4f} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Time: {epoch_time:.2f}s")

    print(f"\nBest Val Dice: {best_val_dice:.4f} at epoch {best_epoch+1}")

    return history, best_val_dice
