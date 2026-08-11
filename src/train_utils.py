"""Shared training utilities: the per-epoch loop, evaluation, and best-epoch checkpointing.

Every model in Part 2 reuses these functions, so the notebook stays free of
copy-pasted training code and all runs share one protocol.
"""

import time

import torch

__all__ = ["train_one_epoch", "evaluate", "train_model"]


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch; return (average loss, top-1 accuracy %)."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += targets.size(0)

    return running_loss / len(dataloader), 100.0 * correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate in eval mode; return (loss, top-1 accuracy %, predictions, targets)."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            running_loss += criterion(outputs, targets).item()

            preds = outputs.argmax(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    return (running_loss / len(dataloader), 100.0 * correct / total,
            all_preds, all_targets)


def train_model(model, trainloader, testloader, criterion, optimizer, scheduler,
                epochs, device, model_name, save_dir=None, save_best=True):
    """Shared training loop: per-epoch evaluation + best-checkpoint persistence.

    Whenever test accuracy improves, the weights are saved to
    ``save_dir / f"{model_name}_best.pth"`` (skipped when ``save_dir`` is None).

    Returns a history dict with losses/accuracies per epoch, the best test accuracy,
    the epoch it was reached at, and the total wall-clock time.
    """
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    best_acc, best_epoch = 0.0, 0

    print(f"\nTraining '{model_name}' on {str(device).upper()} ...")
    print("-" * 66)
    start = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc, _, _ = evaluate(model, testloader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        if save_best and test_acc > best_acc:
            best_acc, best_epoch = test_acc, epoch
            if save_dir is not None:
                torch.save(model.state_dict(), save_dir / f"{model_name}_best.pth")

        if scheduler is not None:
            scheduler.step()

        if epoch in (1, epochs) or epoch % 5 == 0:
            print(f"Epoch {epoch:3d}/{epochs} | Train loss {train_loss:.4f} | "
                  f"Train acc {train_acc:.2f}% | Test acc {test_acc:.2f}%")

    elapsed = time.time() - start
    print("-" * 66)
    print(f"Completed {epochs} epochs in {elapsed:.1f}s  |  "
          f"Best test acc {best_acc:.2f}% at epoch {best_epoch}")

    return {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_losses": test_losses,
        "test_accs": test_accs,
        "best_accuracy": best_acc,
        "best_epoch": best_epoch,
        "training_time": elapsed,
    }
