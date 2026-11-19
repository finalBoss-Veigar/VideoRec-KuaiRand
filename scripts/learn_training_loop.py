"""CPU-only synthetic teaching demo. These numbers are NOT project performance."""
import json
import torch
from torch import nn


def fresh_model():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    return model


def main():
    torch.manual_seed(42)
    model = fresh_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.BCEWithLogitsLoss()
    x, y = torch.tensor([[1.]]), torch.tensor([[1.]])

    optimizer.zero_grad(set_to_none=True)       # Clear previous gradients.
    logit = model(x)                           # Forward: z = w*x.
    loss = criterion(logit, y)                 # Stable sigmoid + binary CE.
    before = {'weight': model.weight.item(), 'logit': logit.item(),
              'probability': logit.sigmoid().item(), 'loss': loss.item()}
    loss.backward()                           # Compute dL/dw; weight unchanged.
    gradient = model.weight.grad.item()
    weight_after_backward = model.weight.item()
    torch.testing.assert_close(model.weight.grad, (logit.detach().sigmoid()-y)*x)
    optimizer.step()                          # Now update w <- w - lr*grad.
    with torch.no_grad():
        after_logit = model(x)
        after = {'weight': model.weight.item(), 'probability': after_logit.sigmoid().item(),
                 'loss': criterion(after_logit, y).item()}

    # Exactly two equal-size microbatches, deterministic model, no dropout.
    large, accumulated = fresh_model(), fresh_model()
    xx = torch.tensor([[1.],[2.]]); yy = torch.ones_like(xx)
    criterion(large(xx), yy).backward()
    for index in range(2):
        (criterion(accumulated(xx[index:index+1]), yy[index:index+1])/2).backward()
    torch.testing.assert_close(large.weight.grad, accumulated.weight.grad)
    result = {'kind': 'synthetic_teaching_only', 'device': 'cpu', 'torch': torch.__version__,
              'before': before, 'gradient': gradient, 'weight_after_backward': weight_after_backward,
              'after_optimizer_step': after,
              'accumulation': {'batch2_gradient': large.weight.grad.item(),
                               'two_microbatches_gradient': accumulated.weight.grad.item(),
                               'scope': 'Exact toy check only; real training equivalence also depends on normalization, randomness, and batch-dependent layers.'}}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
