from autodiff import Tensor
import numpy as np

x = Tensor(np.ones((2, 2)))
y = x + 2
z = y * y * 3
out = z.mean()

print(x)
print(y)
print(z)
print(out)

out.backward()
print(x.grad)

x = Tensor(np.random.randn(3))

y = x * 2
while np.linalg.norm(y.item) < 1000:
    y = y * 2

print(x)
print(y)
gradients = Tensor([0.1, 1.0, 0.0001])
y.backward(gradients)

print(x.grad)

x = Tensor(2)

y = 1 / x
y *= 2
y.backward()
print(x.grad)
