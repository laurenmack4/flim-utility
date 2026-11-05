"""
Module for functionality relating to the fitting of lifetime curves.
"""
import numpy as np

from .FitBounds import SingleExponentialFitBounds, DoubleExponentialFitBounds

from scipy.optimize import curve_fit

# --------------------------------------------------- Constants --------------------------------------------------------
# The factor by which we extend the t-axis when generating IRF-convolved profiles.
T_AXIS_PAD_FACTOR = 5

# The amount by which we roll the decay profile when generating IRF-convolved decay profiles
DECAY_ROLL = 100

# ------------------------------------- Utility functions -------------------------------------------------------------
def _extend_time_axis(time_axis: np.ndarray, factor: float | int = T_AXIS_PAD_FACTOR) -> np.ndarray:
    """
    Extend a time axis by a specified factor.

    @param time_axis: the linear time axis to be extended.
    @param factor: the factor by which to extend the axis.

    @return: the extended time axis.
    """
    dt = time_axis[1] - time_axis[0]
    extended_time_axis = np.linspace(start=0,
                                     stop=factor * len(time_axis) * dt,
                                     num=factor * len(time_axis))
    return extended_time_axis

def _convolve_and_crop_decay(decay: np.ndarray, irf: np.ndarray, length: int) -> np.ndarray:
    """
    Convolve a decay with an IRF and crop to a desired size.

    @param decay: the decay curve.
    @param irf: The IRF to convolve the decay with. Must be normalised. Must be on the same time-binning as the decay.
    @param length: The desired length of the convolved decay.

    @return: the convolved and cropped decay.
    """
    # Roll the decay slightly so that we are sure to capture the top
    decay = np.roll(decay, DECAY_ROLL)

    # Convolve with our IRF
    irf_convolved = np.convolve(decay, irf, mode="full")

    # Finally, we crop down so that the simulated decay matches the input
    irf_convolved = irf_convolved[irf_convolved.argmax():]
    irf_convolved = irf_convolved[:length]

    return irf_convolved


# --------------------------------- Functions describing lifetime decays -----------------------------------------------
def single_exponential_decay(t_axis: np.ndarray, a0: float | int, t0: float, bg: float | int):
    """
    Creates a single exponential decay profile, of the form I(t) = a0 * exp(-t / tau) + bg.

    @param t_axis: the time axis for the decay. Units seconds.
    @param a0: the intensity parameter. Units counts.
    @param t0: the lifetime. Units seconds.
    @param bg: the background parameter. Units counts.
    @return: the specified single-exponential decay.
    """
    return a0 * np.exp(- t_axis / t0) + bg

def bi_exponential_decay(t_axis: np.ndarray, a0: float | int, t0: float, a1: float | int, t1: float, bg: float | int):
    """
    Creates a double-exponential decay profile of the form I(t) = a0 * exp(-t / tau_0) + a1 * exp(-t / tau_1) + bg.

    @param t_axis: the time axis for the decay. Units seconds.
    @param a0: the intensity parameter for the first lifetime component. Units counts.
    @param t0: The lifetime for the first component. Units seconds.
    @param a1: the intensity parameter for the second lifetime component. Units counts.
    @param t1: The lifetime for the second component. Units seconds.
    @param bg: the background parameter. Units counts.
    @return: the specified double-exponential decay.
    """
    return a0 * np.exp(- t_axis / t0) + a1 * np.exp(- t_axis / t1) + bg

# ---------------------------------------- "Curried" functions for lifetime fitting ------------------------------------
def bi_exponential_decay_clamped_t0(t0: float) -> callable:
    """
    Curried function describing a bi-exponential decay where the first lifetime component t0 can be specified.

    The function is designed this way so it can be used with scipy.optimize.curve_fit(), allowing the t0 parameter to
    be specified. The t0 parameter is in scope to the inner function.

    @param t0: the known t0 value
    @return: the inner function describing the double-exponential decay.
    """
    def _bi_exponential_decay_clamped_t0(t_axis: np.ndarray,
                                         a0: float | int,
                                         a1: float | int,
                                         t1: float | int,
                                         bg: float | int):
        """
        Double exponential decay, where the known t0 variable is defined in the outer function.

        @param t_axis: the time axis for the decay. Units seconds.
        @param a0: the intensity parameter for the first lifetime component. Units counts.
        @param a1: the intensity parameter for the second lifetime component. Units counts.
        @param t1: The lifetime for the second component. Units seconds.
        @param bg: the background parameter. Units counts.
        @return: the specified double-exponential decay.
        """
        return bi_exponential_decay(t_axis=t_axis, a0=a0, t0=t0, a1=a1, t1=t1, bg=bg)
    return _bi_exponential_decay_clamped_t0

def _irf_convolved_single_exponential_decay(irf: np.ndarray):
    """
    Curried function describing a single exponential decay that is convolved with an IRF.

    The function is designed this way so it can be used with scipy.optimize.curve_fit(), allowing the IRF to
    be specified. The IRF parameter is in scope to the inner function.

    @param irf: The IRF to convolve the decay with.
    @return: the inner function describing the single-exponential decay.
    """
    def __irf_convolved_single_exponential_decay(t_axis: np.ndarray, a0: float, t0: float, bg: float):
        """
        Single exponential decay, convolved with an IRF, where the IRF is defined in the outer function.

        @param t_axis: the time axis for the decay. Units seconds.
        @param a0: the intensity parameter for the lifetime component. Units counts.
        @param t0: The lifetime for the decay component. Units seconds.
        @param bg: the background parameter. Units counts.
        @return: the specified IRF-convolved single-exponential decay.
        """
        decay = irf_convolved_single_exponential_decay(time_axis=t_axis, irf=irf, a0=a0, t0=t0, bg=bg)
        return decay
    return __irf_convolved_single_exponential_decay

def _irf_convolved_double_exponential_decay(irf: np.ndarray):
    """
    Curried function describing a double exponential decay that is convolved with an IRF.

    The function is designed this way so it can be used with scipy.optimize.curve_fit(), allowing the IRF to
    be specified. The IRF parameter is in scope to the inner function.

    @param irf: The IRF to convolve the decay with.
    @return: the inner function describing the single-exponential decay.
    """
    def __irf_convolved_double_exponential_decay(t_axis: np.ndarray, a0: float, t0: float, a1: float, t1: float, bg: float):
        """
        Double exponential decay, convolved with an IRF, where the IRF is defined in the outer function.

        @param t_axis: the time axis for the decay. Units seconds.
        @param a0: the intensity parameter for the first lifetime component. Units counts.
        @param t0: The lifetime for the first decay component. Units seconds.
        @param a1: the intensity parameter for the second lifetime component. Unit counts.
        @param t1: the lifetime for the second decay component. Unit seconds.
        @param bg: the background parameter. Units counts.
        @return: the specified IRF-convolved single-exponential decay.
        """
        decay = irf_convolved_bi_exponential_decay(time_axis=t_axis, irf=irf, a0=a0, t0=t0, a1=a1, t1=t1, bg=bg)
        return decay

    return __irf_convolved_double_exponential_decay

def _irf_convolved_double_exponential_decay_clamped_t0(irf: np.ndarray, t0: float):
    """
    Curried function describing a double exponential decay that is convolved with an IRF.

    The function is designed this way so it can be used with scipy.optimize.curve_fit(), allowing the IRF and the t0 value
    to be specified. These parameters are in scope to the inner function.

    @param irf: The IRF to convolve the decay with.
    @param t0: the lifetime of the first decay component. Unit seconds.
    @return: the inner function describing the double-exponential decay.
    """
    def __irf_convolved_double_exponential_decay_clamped_t0(t_axis: np.ndarray, a0: float, a1: float, t1: float, bg: float):
        """
        Double exponential decay, convolved with an IRF, where the IRF and the t0 value is defined in the outer function.

        @param t_axis: the time axis for the decay. Units seconds.
        @param a0: the intensity parameter for the first lifetime component. Units counts.
        @param a1: the intensity parameter for the second lifetime component. Unit counts.
        @param t1: the lifetime for the second decay component. Unit seconds.
        @param bg: the background parameter. Units counts.
        @return: the specified IRF-convolved single-exponential decay.
        """
        decay = irf_convolved_bi_exponential_decay(time_axis=t_axis, irf=irf, a0=a0, t0=t0, a1=a1, t1=t1, bg=bg)
        return decay

    return __irf_convolved_double_exponential_decay_clamped_t0

# ----------------------------------------- IRF-convolved decay functions ----------------------------------------------
def irf_convolved_single_exponential_decay(time_axis: np.ndarray,
                                           irf: np.ndarray,
                                           a0: float | int,
                                           t0: float | int,
                                           bg: float | int) -> np.ndarray:
    """
    Generate an IRF-convolved single-exponential decay profile.

    @param time_axis: the time-axis for the decay. Units seconds.
    @param irf: The IRF. Must be normalised. Must be on the same time-binning as the decay.
    @param a0: the intensity parameter. Units counts.
    @param t0: t0: the lifetime. Units seconds.
    @param bg: the background parameter. Units counts.

    @return: the specified IRF-convolved single-exponential decay profile.
    """

    # First, regenerate the time-axis over a longer timescale. This is necessary for the avoidance of any convolution
    # artefacts.
    start_length = len(time_axis)
    time_axis = _extend_time_axis(time_axis)

    # Create the clean decay
    decay = single_exponential_decay(t_axis=time_axis, a0=a0, t0=t0, bg=bg)

    # Convolve with the IRF
    decay = _convolve_and_crop_decay(decay=decay, irf=irf, length=start_length)

    return decay

def irf_convolved_bi_exponential_decay(time_axis: np.ndarray,
                                       irf: np.ndarray,
                                       a0: float | int,
                                       t0: float | int,
                                       a1: float | int,
                                       t1: float| int,
                                       bg : float | int) -> np.ndarray:
    """
    Generate an IRF-convolved double-exponential decay profile.

    @param time_axis: the time-axis for the decay. Units seconds.
    @param irf: The IRF. Must be normalised. Must be on the same time-binning as the decay.
    @param a0: the intensity parameter for the first lifetime component. Units counts.
    @param t0: the lifetime of the first component. Units seconds.
    @param a1: the intensity parameter for the second lifetime component. Units counts.
    @param t1: the lifetime of the second component. Units seconds.
    @param bg: the background parameter. Units counts.

    @return: the specified IRF-convolved double-exponential decay profile.
    """
    # First, regenerate the time-axis over a longer timescale. This is necessary for the avoidance of any convolution
    # artefacts.
    start_length = len(time_axis)
    time_axis = _extend_time_axis(time_axis=time_axis)

    # Create the clean decay
    decay = bi_exponential_decay(t_axis=time_axis, a0=a0, t0=t0, a1=a1, t1=t1, bg=bg)

    # Roll the decay slightly so that we are sure to capture the top
    decay = np.roll(decay, DECAY_ROLL)

    # Convolve with our IRF
    decay = _convolve_and_crop_decay(decay=decay, irf=irf, length=start_length)

    return decay


# ------------------------------------------------ Fitting functions --------------------------------------------------
class FitResult:
    """Immutable object for storing the result of a lifetime fit."""
    def __init__(self,
                 a0: float | int,
                 t0: float | int | None,
                 a1: float | int | None,
                 t1: float | int | None,
                 bg: float | int,
                 a0_uncertainty: float,
                 t0_uncertainty: float | None,
                 a1_uncertainty: float | None,
                 t1_uncertainty: float | None,
                 bg_uncertainty: float,
                 chisq: float | int):
        """
        Constructor for FitResult.

        @param a0: the intensity parameter for the first lifetime component. Units counts.
        @param t0: the lifetime of the first component. Units seconds.
        @param a1: the intensity parameter for the second lifetime component. Units counts.
        @param t1: the lifetime of the second component. Units seconds.
        @param bg: the background count parameter. Units counts.
        @param a0_uncertainty: the uncertainty on the calculated a0 value. Unit counts.
        @param t0_uncertainty: the uncertainty on the calculated t0 value. Unit seconds.
        @param a1_uncertainty: the uncertainty on the calculated a1 value. Unit counts.
        @param t1_uncertainty: the uncertainty on the calculated t1 value. Unit seconds.
        @param bg_uncertainty: the uncertainty on the calculated background value. Unit counts.
        @param chisq: the reduced chi squared for the fit.
        """
        self._a0 = a0
        self._t0 = t0
        self._a1 = a1
        self._t1 = t1
        self._bg = bg
        self._a0_uncertainty = a0_uncertainty
        self._t0_uncertainty = t0_uncertainty
        self._a1_uncertainty = a1_uncertainty
        self._t1_uncertainty = t1_uncertainty
        self._bg_uncertainty = bg_uncertainty
        self._chisq = chisq

    # ---------------------------------------------- Private properties -----------------------------------------------
    @property
    def a0(self) -> float | int:
        """Getter for the a0 parameter."""
        return self._a0 if self._a0 is not None else 0

    @property
    def t0(self) -> float:
        """Getter for the t0 parameter."""
        return  self._t0 if self._t0 is not None else 0

    @property
    def a1(self) -> float | int | None:
        """Getter for the a1 parameter."""
        return self._a1 if self._a1 is not None else 0

    @property
    def t1(self) -> float | int | None:
        """Getter for the t1 parameter."""
        return self._t1 if self._t1 is not None else 0

    @property
    def bg(self) -> float | int:
        """Getter for the background parameter."""
        return self._bg if self._bg is not None else 0

    @property
    def a0_uncertainty(self) -> float:
        """Getter for the uncertainty on the fitted a0 value."""
        return self._a0_uncertainty

    @property
    def t0_uncertainty(self) -> float:
        """Getter for the uncertainty on the fitted t0 value."""
        return self._t0_uncertainty

    @property
    def a1_uncertainty(self) -> float | None:
        """Getter for the uncertainty on the fitted a1 value."""
        return self._a1_uncertainty

    @property
    def t1_uncertainty(self) -> float | None:
        """Getter for the uncertainty on the fitted t1 value."""
        return self._t1_uncertainty

    @property
    def bg_uncertainty(self) -> float:
        """Getter for the uncertainty on the fitted bg value."""
        return self._bg_uncertainty

    @property
    def chisq(self) -> float:
        """Getter for the reduced chi-squared of the fit."""
        return self._chisq

def fit_single_exponential(histogram: np.ndarray, t_bin_width: float, bounds: SingleExponentialFitBounds) -> FitResult:
    """
    Fit a single exponential decay (not IRF convolved) to a decay histogram.

    @param histogram: the histogram we are fitting to.
    @param t_bin_width: the size of each time bin. Units seconds.
    @param bounds: the bounds on each parameter being fitted.

    @return: the least-squares best-fit.
    """

    # First, we need to get the bounds into the format that is accepted by scipy.curve_fit().
    lower_bounds = (bounds.a0_bounds.lower, bounds.t0_bounds.lower, bounds.bg_bounds.lower)
    upper_bounds = (bounds.a0_bounds.upper, bounds.t0_bounds.upper, bounds.bg_bounds.upper)


    t_axis = np.linspace(0, len(histogram) * t_bin_width, len(histogram))

    # Calculate the uncertainty
    uncertainty = np.sqrt(np.maximum(histogram, 1))

    # Do the fitting
    fit = curve_fit(f=single_exponential_decay,
                     xdata=t_axis,
                     ydata=histogram,
                     bounds=(lower_bounds, upper_bounds),
                     sigma=uncertainty, absolute_sigma=True)

    lsq_sol = np.float64(fit[0])
    covariance = np.float64(fit[1])
    fit_uncertainty = np.sqrt(np.diag(covariance))

    regen = single_exponential_decay(t_axis=t_axis, a0=lsq_sol[0], t0=lsq_sol[1], bg=lsq_sol[2])
    reduced_chi_squared = np.sum(((histogram - regen) / uncertainty)**2) / (len(regen) - 3) # n params = 3 for single exponential fit

    res = FitResult(a0=lsq_sol[0],
                    t0=lsq_sol[1],
                    a1=None, t1=None,
                    bg=lsq_sol[2],
                    a0_uncertainty=fit_uncertainty[0],
                    t0_uncertainty=fit_uncertainty[1],
                    a1_uncertainty=None,
                    t1_uncertainty=None,
                    bg_uncertainty=fit_uncertainty[2],
                    chisq=reduced_chi_squared)

    return res

def fit_double_exponential(histogram: np.ndarray, t_bin_width: float, bounds: DoubleExponentialFitBounds) -> FitResult:
    """
    Fit a double exponential decay (not IRF convolved) to a decay histogram.

    @param histogram: the histogram we are fitting to.
    @param t_bin_width: the size of each time bin. Units seconds.
    @param bounds: the bounds on each parameter being fitted.

    @return: the least-squares best-fit.
    """
    # First, we need to get the bounds into the format that is accepted by scipy.curve_fit().
    lower_bounds = (bounds.a0_bounds.lower, bounds.t0_bounds.lower, bounds.a1_bounds.lower, bounds.t1_bounds.lower, bounds.bg_bounds.lower)
    upper_bounds = (bounds.a0_bounds.upper, bounds.t0_bounds.upper, bounds.a1_bounds.upper, bounds.t1_bounds.upper, bounds.bg_bounds.upper)

    # Crop the histogram
    t_axis = np.linspace(0, len(histogram) * t_bin_width, len(histogram))

    # Calculate the uncertainty
    uncertainty = np.sqrt(np.maximum(histogram, 1))

    # Do the fitting
    fit = curve_fit(f=bi_exponential_decay,
                    xdata=t_axis,
                    ydata=histogram,
                    bounds=(lower_bounds, upper_bounds),
                    sigma=uncertainty, absolute_sigma=True)

    lsq_sol = np.float64(fit[0])
    covariance = np.float64(fit[1])
    fit_uncertainty = np.sqrt(np.diag(covariance))

    regen = bi_exponential_decay(t_axis=t_axis, a0=lsq_sol[0], t0=lsq_sol[1], a1=lsq_sol[2], t1=lsq_sol[3], bg=lsq_sol[4])
    reduced_chi_squared = np.sum(((histogram - regen) / uncertainty) ** 2) / (
                len(regen) - 5)  # n params = 5 for a double exponential fit

    res = FitResult(a0=lsq_sol[0],
                    t0=lsq_sol[1],
                    a1=lsq_sol[2],
                    t1=lsq_sol[3],
                    bg=lsq_sol[4],
                    a0_uncertainty=fit_uncertainty[0],
                    t0_uncertainty=fit_uncertainty[1],
                    a1_uncertainty=fit_uncertainty[2],
                    t1_uncertainty=fit_uncertainty[3],
                    bg_uncertainty=fit_uncertainty[4],
                    chisq=reduced_chi_squared)

    return res


def fit_double_exponential_clamped_t0(histogram: np.ndarray,
                                      t_bin_width: float,
                                      bounds: DoubleExponentialFitBounds,
                                      t0: float) -> FitResult:
    """
    Fit a double exponential decay (not IRF convolved) to a decay histogram, where the first lifetime component t0
    is known (and specified)

    @param histogram: the histogram we are fitting to.
    @param t_bin_width: the size of each time bin. Units seconds.
    @param bounds: the bounds on each parameter being fitted.
    @param t0: the known first lifetime.

    @return: the least-squares best-fit.
    """
    # First, we need to get the bounds into the format that is accepted by scipy.curve_fit().
    lower_bounds = (bounds.a0_bounds.lower, bounds.a1_bounds.lower, bounds.t1_bounds.lower, bounds.bg_bounds.lower)
    upper_bounds = (bounds.a0_bounds.upper, bounds.a1_bounds.upper, bounds.t1_bounds.upper, bounds.bg_bounds.upper)

    # Crop the histogram
    hist_cropped = histogram[histogram.argmax():].copy()
    t_axis_cropped = np.linspace(0, len(hist_cropped) * t_bin_width, len(hist_cropped))

    # Calculate the uncertainty
    uncertainty = np.sqrt(np.maximum(hist_cropped, 1))

    # Do the fitting
    fit = curve_fit(f=bi_exponential_decay_clamped_t0(t0=t0),
                    xdata=t_axis_cropped,
                    ydata=hist_cropped,
                    bounds=(lower_bounds, upper_bounds),
                    sigma=uncertainty, absolute_sigma=True)
    lsq_sol = np.float64(fit[0])
    covariance = np.float64(fit[1])
    fit_uncertainty = np.sqrt(np.diag(covariance))

    regen = bi_exponential_decay(t_axis=t_axis_cropped, a0=lsq_sol[0], t0=t0, a1=lsq_sol[1], t1=lsq_sol[2],
                                 bg=lsq_sol[3])
    reduced_chi_squared = np.sum(((hist_cropped - regen) / uncertainty) ** 2) / (
            len(regen) - 4)  # n params = 4 for a double exponential fit with t0 clamped

    res = FitResult(a0=lsq_sol[0],
                    t0=t0,
                    a1=lsq_sol[1],
                    t1=lsq_sol[2],
                    bg=lsq_sol[3],
                    a0_uncertainty=fit_uncertainty[0],
                    t0_uncertainty=None,
                    a1_uncertainty=fit_uncertainty[1],
                    t1_uncertainty=fit_uncertainty[2],
                    bg_uncertainty=fit_uncertainty[3],
                    chisq=reduced_chi_squared)

    return res

def fit_irf_convolved_single_exponential(histogram: np.ndarray,
                                         irf: np.ndarray,
                                         t_bin_width: float,
                                         bounds: SingleExponentialFitBounds,
                                         ) -> FitResult:
    """
    Fit a single exponential decay that is convolved with an IRF to a decay histogram.

    @param histogram: the histogram we are fitting to.
    @param t_bin_width: the size of each time bin. Units seconds.
    @param bounds: the bounds on each parameter being fitted.
    @param irf: the IRF corresponding to the measured histogram. Must have the same time-domain binning.

    @return: the least-squares best-fit.
    """
    # First, we need to get the bounds into the format that is accepted by scipy.curve_fit().
    lower_bounds = (bounds.a0_bounds.lower, bounds.t0_bounds.lower, bounds.bg_bounds.lower)
    upper_bounds = (bounds.a0_bounds.upper, bounds.t0_bounds.upper, bounds.bg_bounds.upper)

    # Crop the histogram
    hist_cropped = histogram[histogram.argmax():].copy()
    t_axis_cropped = np.linspace(0, len(hist_cropped) * t_bin_width, len(hist_cropped))

    # Calculate the uncertainty
    uncertainty = np.sqrt(np.maximum(hist_cropped, 1))

    # Do the fitting
    fit = curve_fit(f=_irf_convolved_single_exponential_decay(irf=irf),
                    xdata=t_axis_cropped,
                    ydata=hist_cropped,
                    bounds=(lower_bounds, upper_bounds),
                    sigma=uncertainty, absolute_sigma=True)

    lsq_sol = np.float64(fit[0])
    covariance = np.float64(fit[1])
    fit_uncertainty = np.sqrt(np.diag(covariance))

    regen = irf_convolved_single_exponential_decay(time_axis=t_axis_cropped, irf=irf, a0=lsq_sol[0], t0=lsq_sol[1],  bg=lsq_sol[2])

    reduced_chi_squared = np.sum(((hist_cropped - regen) / uncertainty) ** 2) / (len(regen) - 3)  # n params = 3 for a single exponential fit

    res = FitResult(a0=lsq_sol[0],
                    t0=lsq_sol[1],
                    a1=None,
                    t1=None,
                    bg=lsq_sol[2],
                    a0_uncertainty=fit_uncertainty[0],
                    t0_uncertainty=fit_uncertainty[1],
                    a1_uncertainty=None,
                    t1_uncertainty=None,
                    bg_uncertainty=fit_uncertainty[2],
                    chisq=reduced_chi_squared)

    return res


def fit_irf_convolved_double_exponential(histogram: np.ndarray,
                                         irf: np.ndarray,
                                         t_bin_width: float,
                                         bounds: DoubleExponentialFitBounds,
                                         ) -> FitResult:
    """
    Fit a double exponential decay that is convolved with an IRF to a decay histogram.

    @param histogram: the histogram we are fitting to.
    @param t_bin_width: the size of each time bin. Units seconds.
    @param bounds: the bounds on each parameter being fitted.
    @param irf: the IRF corresponding to the measured histogram. Must have the same time-domain binning.

    @return: the least-squares best-fit.
    """
    # First, we need to get the bounds into the format that is accepted by scipy.curve_fit().
    lower_bounds = (bounds.a0_bounds.lower, bounds.t0_bounds.lower, bounds.a1_bounds.lower, bounds.t1_bounds.lower, bounds.bg_bounds.lower)
    upper_bounds = (bounds.a0_bounds.upper, bounds.t0_bounds.upper, bounds.a1_bounds.upper, bounds.t1_bounds.upper, bounds.bg_bounds.upper)

    # Crop the histogram
    hist_cropped = histogram[histogram.argmax():].copy()
    t_axis_cropped = np.linspace(0, len(hist_cropped) * t_bin_width, len(hist_cropped))

    # Calculate the uncertainty
    uncertainty = np.sqrt(np.maximum(hist_cropped, 1))

    # Do the fitting
    fit = curve_fit(f=_irf_convolved_double_exponential_decay(irf=irf),
                    xdata=t_axis_cropped,
                    ydata=hist_cropped,
                    bounds=(lower_bounds, upper_bounds),
                    sigma=uncertainty, absolute_sigma=True)

    lsq_sol = np.float64(fit[0])
    covariance = np.float64(fit[1])
    fit_uncertainty = np.sqrt(np.diag(covariance))

    regen = irf_convolved_bi_exponential_decay(time_axis=t_axis_cropped, irf=irf, a0=lsq_sol[0], t0=lsq_sol[1], a1=lsq_sol[2], t1=lsq_sol[3],
                                                   bg=lsq_sol[4])

    reduced_chi_squared = np.sum(((hist_cropped - regen) / uncertainty) ** 2) / (
                len(regen) - 5)  # n params = 5 for a double exponential fit

    res = FitResult(a0=lsq_sol[0],
                    t0=lsq_sol[1],
                    a1=lsq_sol[2],
                    t1=lsq_sol[3],
                    bg=lsq_sol[4],
                    a0_uncertainty=fit_uncertainty[0],
                    t0_uncertainty=fit_uncertainty[1],
                    a1_uncertainty=fit_uncertainty[2],
                    t1_uncertainty=fit_uncertainty[3],
                    bg_uncertainty=fit_uncertainty[4],
                    chisq=reduced_chi_squared)

    return res


def fit_irf_convolved_double_exponential_clamped_t0(histogram: np.ndarray,
                                                    irf: np.ndarray,
                                                    t_bin_width: float,
                                                    bounds: DoubleExponentialFitBounds,
                                                    t0: float
                                                    ) -> FitResult:
    """
    Fit a double exponential decay that is convolved with an IRF to a decay histogram, where the first lifetime
    component is known.

    @param histogram: the histogram we are fitting to.
    @param t_bin_width: the size of each time bin. Units seconds.
    @param bounds: the bounds on each parameter being fitted.
    @param t0: the value of the first lifetime component. Unit seconds.
    @param irf: the IRF corresponding to the measured histogram. Must have the same time-domain binning.

    @return: the least-squares best-fit.
    """
    # First, we need to get the bounds into the format that is accepted by scipy.curve_fit().
    lower_bounds = (bounds.a0_bounds.lower, bounds.a1_bounds.lower, bounds.t1_bounds.lower, bounds.bg_bounds.lower)
    upper_bounds = (bounds.a0_bounds.upper, bounds.a1_bounds.upper, bounds.t1_bounds.upper, bounds.bg_bounds.upper)

    # Crop the histogram
    hist_cropped = histogram[histogram.argmax():].copy()
    t_axis_cropped = np.linspace(0, len(hist_cropped) * t_bin_width, len(hist_cropped))

    # Calculate the uncertainty
    uncertainty = np.sqrt(np.maximum(hist_cropped, 1))

    # Do the fitting
    fit = curve_fit(f=_irf_convolved_double_exponential_decay_clamped_t0(irf=irf, t0=t0),
                    xdata=t_axis_cropped,
                    ydata=hist_cropped,
                    bounds=(lower_bounds, upper_bounds),
                    sigma=uncertainty, absolute_sigma=True)

    lsq_sol = np.float64(fit[0])
    covariance = np.float64(fit[1])
    fit_uncertainty = np.sqrt(np.diag(covariance))

    regen = irf_convolved_bi_exponential_decay(time_axis=t_axis_cropped, irf=irf, a0=lsq_sol[0], t0=t0, a1=lsq_sol[1], t1=lsq_sol[2],
                                                   bg=lsq_sol[3])

    reduced_chi_squared = np.sum(((hist_cropped - regen) / uncertainty) ** 2) / (
                len(regen) - 4)  # n params = 4 for a double exponential fit where t0 is known

    res = FitResult(a0=lsq_sol[0],
                    t0=t0,
                    a1=lsq_sol[1],
                    t1=lsq_sol[2],
                    bg=lsq_sol[3],
                    a0_uncertainty=fit_uncertainty[0],
                    t0_uncertainty=None,
                    a1_uncertainty=fit_uncertainty[1],
                    t1_uncertainty=fit_uncertainty[2],
                    bg_uncertainty=fit_uncertainty[3],
                    chisq=reduced_chi_squared)

    return res