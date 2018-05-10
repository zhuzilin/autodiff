import uuid

class Operator:
    def __init__(self, *args, **kwargs):
        self._id = uuid.uuid4()
        self._inputs = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        args = list(args)
        for i in range(len(args)):
            args[i].add_out()
        self._inputs = args

    def backward(self, grad_out):
        raise NotImplementedError

    def zero_grad(self):
        for i in self._inputs:
            i.zero_grad()

