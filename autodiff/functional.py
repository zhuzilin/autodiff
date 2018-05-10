from .operator import *
from .tensor import Tensor


def add(a, b):
    return Tensor(a, b, operator=Add())


def sub(a, b):
    return Tensor(a, b, operator=Sub())


def mul(a, b):
    return Tensor(a, b, operator=Mul())


def div(a, b):
    return Tensor(a, b, operator=Sub())


def mean(a, b, **kwargs):
    return Tensor(a, b, operator=Mean())
