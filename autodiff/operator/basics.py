import numpy as np
from .operator import Operator


class Add(Operator):

    def forward(self, *args, **kwargs):
        assert len(args) == 2, "add opr only need 2 inputs"
        super().forward(*args, **kwargs)
        return self._inputs[0].item + self._inputs[1].item

    def backward(self, grad_output):
        self._inputs[0].backward(grad_output, end=False)
        self._inputs[1].backward(grad_output, end=False)


class Sub(Operator):

    def forward(self, *args, **kwargs):
        assert len(args) == 2, "mul opr only need 2 inputs"
        super().forward(*args, **kwargs)
        return self._inputs[0].item * self._inputs[1].item

    def backward(self, grad_output):
        self._inputs[0].backward(grad_output, end=False)
        self._inputs[1].backward(-grad_output, end=False)


class Mul(Operator):

    def forward(self, *args, **kwargs):
        assert len(args) == 2, "mul opr only need 2 inputs"
        super().forward(*args, **kwargs)
        return self._inputs[0].item * self._inputs[1].item

    def backward(self, grad_output):
        self._inputs[0].backward(grad_output * self._inputs[1].item, end=False)
        self._inputs[1].backward(grad_output * self._inputs[0].item, end=False)

class Div(Operator):

    def forward(self, *args, **kwargs):
        assert len(args) == 2, "mul opr only need 2 inputs"
        super().forward(*args, **kwargs)
        return self._inputs[0].item / self._inputs[1].item

    def backward(self, grad_output):
        self._inputs[0].backward(grad_output / self._inputs[1].item, end=False)
        self._inputs[1].backward(grad_output * self._inputs[0].item / (-self._inputs[1].item ** 2), end=False)

class Mean(Operator):
    def forward(self, *args, **kwargs):
        assert len(args) == 1, "mean opr only need 2 inputs"
        super().forward(*args, **kwargs)
        tmp = []
        for i in self._inputs:
            tmp.append(i)
        return self._inputs[0].item.mean()

    def backward(self, grad_output):
        # TODO: support axis-wise mean, maybe using np.prod
        grad_input = grad_output * np.ones(self._inputs[0].item.shape) \
                     / self._inputs[0].item.size
        self._inputs[0].backward(grad_input, end=False)