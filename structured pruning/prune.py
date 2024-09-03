import torch.nn.utils.prune as prune
import torch

class StandardPruner:
    def __init__(self, model, prune_threshold=0.01):
        self.model = model
        self.prune_threshold = prune_threshold

    def apply_pruning(self):
        pruned_layers = 0  # Counter for pruned layers
        for module in self.model.modules():
            if hasattr(module, 'weight'):
                prune.l1_unstructured(module, name='weight', amount=self.prune_threshold)
                prune.remove(module, 'weight')
                pruned_layers += 1
                print(f"Pruned {self.prune_threshold * 100}% weights in layer {module}")
        print(f"Total pruned layers: {pruned_layers}")

class StructuredPruner:
    def __init__(self, model, amount=0.5):
        self.model = model
        self.amount = amount

    def apply_pruning(self):
        pruned_layers = 0  # Counter for pruned layers
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.ln_structured(module, name='weight', amount=self.amount, n=1, dim=0)
                prune.remove(module, 'weight')
                pruned_layers += 1
                print(f"Pruned {self.amount * 100}% weights in layer {module}")
        print(f"Total pruned layers: {pruned_layers}")
