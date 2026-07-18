"""
Stage 1: a scalar autograd engine.

A `Value` wraps a single number and remembers how it was computed, so that
calling `.backward()` can walk the resulting graph in reverse and apply the
chain rule at every step.
"""

import math


class Value:
    """A single scalar and the graph edge that produced it."""

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, power):
        out = Value(self.data ** power, (self,), '**')
        def _backward():
            self.grad += (power * self.data ** (power - 1)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), "tanh")
        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        pass

    def __sub__(self, other):
        pass

    def __radd__(self, other):
        pass

    def __rmul__(self, other):
        pass

    def __rsub__(self, other):
        pass

    def __truediv__(self, other):
        pass

    def __rtruediv__(self, other):
        pass

    def backward(self):
        # TODO: build a reverse topological order (DFS, parents before
        # self, then append self), seed self.grad = 1.0, then walk the
        # reversed order calling each node's _backward().
        pass

    def __repr__(self):
        pass


if __name__ == "__main__":
    # TODO: a small worked example you can verify by hand, e.g.
    # d = a*b + c, f = tanh(d), then f.backward() and print the gradients.
    pass
