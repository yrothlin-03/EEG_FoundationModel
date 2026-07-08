import numpy as np


class Metrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.loss_sum = 0.0
        self.count = 0
        self.preds = []
        self.targets = []

    def update(self, logits, targets, loss):
        self.loss_sum += float(loss)
        self.count += 1
        self.preds.append(logits.argmax(-1).detach().cpu().numpy())
        self.targets.append(targets.detach().cpu().numpy())

    @property
    def running_loss(self):
        return self.loss_sum / max(1, self.count)

    def compute(self):
        preds = np.concatenate(self.preds)
        targets = np.concatenate(self.targets)
        acc = float((preds == targets).mean())
        baccs = []
        for c in range(self.num_classes):
            mask = targets == c
            if mask.sum() > 0:
                baccs.append((preds[mask] == c).mean())
        bacc = float(np.mean(baccs)) if baccs else 0.0
        return {"loss": self.running_loss, "ACC": acc, "BACC": bacc}

    @staticmethod
    def format(metrics):
        return " | ".join(f"{k}: {float(v):.4f}" for k, v in metrics.items())
