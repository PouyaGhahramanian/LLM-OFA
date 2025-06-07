import torch
from torch.optim.optimizer import Optimizer, required


class Adaptimizer(Optimizer):
    def __init__(self, params, lr=required, window_size=10, eps=1e-8, weight_decay=0,
                 momentum=0, centered=False, slow_update_rate=0.1):
        defaults = dict(
            lr=lr,
            window_size=window_size,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
            slow_update_rate=slow_update_rate
        )
        super().__init__(params, defaults)

        self.slow_weights = {}
        self.mid_weights = {}
        self.snapshot_buffer = []

        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    pid = id(p)
                    self.slow_weights[pid] = p.data.clone().to(p.device)
                    self.mid_weights[pid] = p.data.clone().to(p.device)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            alpha = 1 / group['window_size']
            eps = group['eps']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            lr = group['lr']
            slow_update_rate = group['slow_update_rate']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adaptimizer does not support sparse gradients')

                state = self.state[p]
                if len(state) == 0:
                    state['square_avg'] = torch.zeros_like(p.data)
                    if momentum > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                square_avg.mul_(1 - alpha).addcmul_(grad, grad, value=alpha)
                avg = square_avg.sqrt().add_(eps)

                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                if momentum > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).addcdiv_(grad, avg)
                    p.data.add_(buf, alpha=-lr)
                else:
                    p.data.addcdiv_(grad, avg, value=-lr)

                pid = id(p)
                slow = self.slow_weights[pid].to(p.device)
                slow += slow_update_rate * (p.data - slow)
                self.slow_weights[pid] = slow.clone()
                self.mid_weights[pid] = 0.5 * (p.data + self.slow_weights[pid])

        return loss

    def sync_slow_weights(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.slow_weights[id(p)] = p.data.clone().to(p.device)
                    self.mid_weights[id(p)] = p.data.clone().to(p.device)

    def apply_weights(self, weight_type='fast'):
        if weight_type not in ['fast', 'slow', 'mid']:
            raise ValueError("weight_type must be 'fast', 'slow', or 'mid'")

        self.fast_weights_backup = {}
        source_weights = {
            'slow': self.slow_weights,
            'mid': self.mid_weights,
            'fast': None  # Do nothing
        }

        for group in self.param_groups:
            for p in group['params']:
                pid = id(p)
                self.fast_weights_backup[pid] = p.data.clone()
                if weight_type != 'fast':
                    p.data.copy_(source_weights[weight_type][pid].to(p.device))

    def restore_fast_weights(self):
        for group in self.param_groups:
            for p in group['params']:
                pid = id(p)
                if pid in self.fast_weights_backup:
                    p.data.copy_(self.fast_weights_backup[pid])
        self.fast_weights_backup = {}

    def get_mid_weights(self):
        return self.mid_weights

    def get_snapshot_buffer(self):
        return self.snapshot_buffer

    def set_learning_rate(self, new_lr):
        for group in self.param_groups:
            group['lr'] = new_lr

    def set_window_size(self, new_size):
        for group in self.param_groups:
            group['window_size'] = new_size

    def compute_drift_score(self, metric='l2'):
        """
        Compute drift between fast and slow weights.

        metric: 'l2' | 'cosine' | 'mad'
        """
        total_score = 0.0
        count = 0

        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                pid = id(p)
                slow = self.slow_weights[pid].to(p.device)
                fast = p.data

                if metric == 'l2':
                    score = torch.norm(fast - slow, p=2).item()
                elif metric == 'mad':
                    score = torch.mean(torch.abs(fast - slow)).item()
                elif metric == 'cosine':
                    score = 1 - torch.nn.functional.cosine_similarity(
                        fast.view(1, -1), slow.view(1, -1)
                    ).item()
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                total_score += score
                count += 1

        return total_score / count if count > 0 else 0.0