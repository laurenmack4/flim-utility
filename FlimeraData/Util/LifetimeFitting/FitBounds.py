from abc import ABC, abstractmethod

class Bound:
    """Object representing the bounds on a parameter being fitted."""
    def __init__(self, lower: float, upper: float):
        """Constructor for Bound.

        :param lower: the lower bound on the parameter being fitted.
        :param upper: the upper bound on the parameter being fitted.
        """
        if lower is not None and upper is not None:
            assert lower < upper, "Lower bound must be strictly less than the upper bound!"

        self._lower = lower
        self._upper = upper

    @property
    def lower(self) -> float:
        """Getter for the lower bound."""
        return self._lower

    @property
    def upper(self) -> float:
        """Getter for the upper bound."""
        return self._upper


class FitBounds(ABC):
    """
    Abstract class for bounds on parameters being fitted during a lifetime calculation.
    """
    def __init__(self,
                 a0_bounds: Bound,
                 t0_bounds: Bound,
                 a1_bounds: Bound | None,
                 t1_bounds: Bound | None,
                 bg_bounds: Bound):
        """
        Constructor for FitBounds.

        @param a0_bounds: Bounds on the a0 parameter.
        @param t0_bounds: Bounds on the t0 parameter.
        @param a1_bounds: Bounds on the a1 parameter.
        @param t1_bounds: Bounds on the t1 parameter.
        @param bg_bounds: Bounds on the background parameter.
        """
        self._a0_bounds = a0_bounds
        self._t0_bounds = t0_bounds
        self._a1_bounds = a1_bounds
        self._t1_bounds = t1_bounds
        self._bg_bounds = bg_bounds

    # ----------------------------------------- Abstract private properties -------------------------------------------
    @property
    def a0_bounds(self) -> Bound:
        """Getter for the bounds on a0."""
        return self._a0_bounds

    @property
    def t0_bounds(self) -> Bound:
        """Getter for the bounds on t0."""
        return self._t0_bounds

    @property
    def bg_bounds(self) -> Bound:
        """Getter for the bounds on the background parameter."""
        return self._bg_bounds

    #---------------------------------------- Non-abstract private properties ------------------------------------------
    @property
    @abstractmethod
    def a1_bounds(self) -> Bound | None: """Getter for the bounds on a1"""

    @property
    @abstractmethod
    def t1_bounds(self) -> Bound | None: """Getter for the bounds on t1."""


class SingleExponentialFitBounds(FitBounds):
    """
    Bounds on parameters being fitted during a single-exponential decay lifetime fit.
    """
    def __init__(self, a0_bounds, t0_bounds, bg_bounds):
        super().__init__(a0_bounds=a0_bounds, t0_bounds=t0_bounds, a1_bounds=None, t1_bounds=None, bg_bounds=bg_bounds)

    # ------------------------------------------ Non-abstract private properties ---------------------------------------
    @property
    def a1_bounds(self) -> Bound | None:
        raise AttributeError(f"a1 is not defined for a single exponential fit!")

    @property
    def t1_bounds(self) -> Bound | None:
        raise AttributeError("t1 is not defined for a single exponential fit!")


class DoubleExponentialFitBounds(FitBounds):
    """
    Bounds on parameters being fitted during a double-exponential decay lifetime fit.
    """
    def __init__(self, a0_bounds, t0_bounds, a1_bounds, t1_bounds, bg_bounds):
        super().__init__(a0_bounds=a0_bounds,
                         t0_bounds=t0_bounds,
                         a1_bounds=a1_bounds,
                         t1_bounds=t1_bounds,
                         bg_bounds=bg_bounds)

    # ------------------------------------------ Non-abstract private properties ---------------------------------------
    @property
    def a1_bounds(self) -> Bound | None:
        return self._a1_bounds

    @property
    def t1_bounds(self) -> Bound | None:
        return self._t1_bounds