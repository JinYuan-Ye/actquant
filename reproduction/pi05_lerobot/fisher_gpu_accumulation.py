"""Equivalent eager Fisher accumulation on GPU, avoiding per-sample D2H copies."""
import torch


class GPUAccumulatorMap(dict):
    """Keep normal dictionary serialization; transfer once when export pops a target."""
    def pop(self, name, *default):
        value = super().pop(name, *default)
        return value.cpu() if isinstance(value, torch.Tensor) else value


def initialize(targets):
    return GPUAccumulatorMap({name: torch.zeros(parameter.shape, dtype=torch.float32, device=parameter.device)
                              for _, name, parameter, _ in targets})


def step(model, targets, accumulators, obs, actions):
    # Same B=1, mean32 loss, autograd targets, FP32 square, and addition order as upstream.
    per_element = model(obs, actions)
    loss = per_element.mean()
    gradients = torch.autograd.grad(loss, [parameter for _, _, parameter, _ in targets],
                                    retain_graph=False, create_graph=False, allow_unused=True)
    for (_, name, _, _), gradient in zip(targets, gradients):
        if gradient is None:
            continue
        if accumulators[name].device != gradient.device:
            accumulators[name] = accumulators[name].to(gradient.device)
        accumulators[name] += gradient.detach().float().pow(2)
    return float(loss.item())
