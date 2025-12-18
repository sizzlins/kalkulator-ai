import sympy as sp


class log2(sp.Function):
    """Base-2 logarithm function."""

    @classmethod
    def eval(cls, arg):
        return sp.log(arg, 2)


class log10(sp.Function):
    """Base-10 logarithm function."""

    @classmethod
    def eval(cls, arg):
        return sp.log(arg, 10)
