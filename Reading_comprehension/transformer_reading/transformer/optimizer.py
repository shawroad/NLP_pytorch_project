from config import Config

class TransformerOptimizer(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, warmup_steps=4000):
        self.optimizer = optimizer
        self.init_lr = Config.d_model ** (-0.5)
        self.warmup_steps = warmup_steps
        self.lr = self.init_lr
        self.step_num = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._update_lr()
        self.optimizer.step()

    def _update_lr(self):
        self.step_num += 1
        self.lr = self.init_lr * min(self.step_num ** (-0.65), self.step_num * (self.warmup_steps ** (-1.5)))
        self.lr = max(self.lr, 1e-5)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
