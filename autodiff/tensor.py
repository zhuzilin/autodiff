import uuid
from numbers import Number
import numpy as np
from .operator.basics import *


class Tensor():
    """Class for Tensor, the basic computing unit in autodiff
    
       Unlike Operator, Tensor can only have one input.
    """
    def __init__(self, *args, **kwargs):
        self._id = uuid.uuid4()
        if 'operator' in kwargs:  # the args is opr instead of
            new_args = []  # use new_args to avoid change the input value
            for i in args:
                if not isinstance(i, Tensor):
                    new_args.append(Tensor(i))
                else:
                    new_args.append(i)
            self._input = kwargs['operator']
            self._item = self._input.forward(*new_args, **kwargs)  # immediately calculate
        else:
            self._input = None
            if isinstance(args[0], Tensor):
                self._item = args[0].item
            elif isinstance(args[0], Number):
                self._item = np.array([args[0]], dtype=np.float32)
            else:
                self._item = np.array(args[0])
        self._grad = np.zeros(self._item.shape)
        self._cnt_out = 0

    def __str__(self):
        return self._item.__str__()

    def add_out(self):
        self._cnt_out += 1

    def zero_grad(self):
        self._grad = np.zeros(self._item.shape)  # Since immediately calculate, don't need to worry if _item is None
        if self._input is not None:
            self._input.zero_grad()

    def backward(self, grad=1., end=True):
        if isinstance(grad, Tensor):
            self.backward(grad.item, end=end)
            return
        # TODO: deal with broadcasting
        if not isinstance(grad, np.ndarray) or self._grad.size >= grad.size:
            self._grad += + grad
        else:
            self._grad = self._grad + grad
        self._cnt_out -= 1
        if self._input is not None and (end or self._cnt_out == 0):
            self._input.backward(self._grad)

    @property
    def item(self):
        return self._item

    @property
    def grad(self):
        return self._grad

    def __add__(self, other):
        return Tensor(self, other, operator=Add())

    def __radd__(self, other):
        return Tensor(self, other, operator=Add())

    def __iadd__(self, other):
        return Tensor(self, other, operator=Add())

    def __sub__(self, other):
        return Tensor(other, self, operator=Sub())

    def __rsub__(self, other):
        return Tensor(self, other, operator=Sub())

    def __isub__(self, other):
        return Tensor(self, other, operator=Sub())

    def __mul__(self, other):
        return Tensor(self, other, operator=Mul())

    def __rmul__(self, other):
        return Tensor(self, other, operator=Mul())

    def __imul__(self, other):
        return Tensor(self, other, operator=Mul())

    def __truediv__(self, other):
        return Tensor(self, other, operator=Div())

    def __rtruediv__(self, other):
        return Tensor(other, self, operator=Div())

    def __itruediv__(self, other):
        return Tensor(other, self, operator=Div())

    def mean(self):
        return Tensor(self, operator=Mean())

    def __le__(self, other):
        return self.item < Tensor(other).item


