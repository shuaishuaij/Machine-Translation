# -*- coding=utf-8 -*-
from omnibox.tools import *
from omnibox.Torch import *


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# >> | Optimizer | <<
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


class ScheduledOptim():
    """A simple wrapper class for learning rate scheduling
        根据步数不同调整lr
        根据层不同调整lr
    """

    def __init__(self, optimizer, d_model, n_warm_up_steps):
        self._optimizer = optimizer
        self.n_warm_up_steps = n_warm_up_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        """Step with the inner optimizer"""
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer
            全模型参数梯度清零
        """
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warm_up_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr



























