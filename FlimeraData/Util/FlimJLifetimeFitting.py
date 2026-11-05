"""
Module that handles all lifetime fitting.

This module mostly wraps the FlimJ plugin and exposes its core functionality through a python interface. The parameters
are intended to be named identically to those in the FlimJ plugin. All the bridging to Java and communication with
imageJ is done internally, so the fitting can be performed externally using pure python.

Note: the module uses pyimagej instead of the python wrapper for the binary files because (i) it seems that the fitting
is much faster this way, (ii) documentation for the python wrapper is sparse, and (iii) I found installing flimlib to
be very non-trivial. This does, however, mean that there are a few prerequisites, as detailed below.

Prerequisites
-------------
* There must be a local installation of Fiji available, with both the FlimJ plugin and FlimOps installed. Both of these
can be installed using the internal update manager, details for FlimJ are here: https://imagej.net/plugins/flimj/ and
details for FlimOps are here: https://github.com/flimlib/flimj-ops/tree/master

* There must be a suitable installation of the Java Development Kit (JDK). Tested with v11.0.15.

* On Windows, the JAVA_HOME environment variable must be set correctly.
"""

from enum import Enum

from loguru import logger
import imagej
import time
import numpy as np
from scipy import ndimage

from .TCSPCImage import TCSPCImage

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib_scalebar.scalebar import ScaleBar

class NoiseModel(Enum):
    """
    Python Enum that mirrors the noise models available in FlimJ.
    """
    GAUSSIAN_FIT = 0,
    POISSON_FIT = 1,
    POISSON_DATA = 2,
    MLE=3

class Algorithm(Enum):
    """
    Python Enum that mirrors the fitting algorithms available in FlimJ.
    """
    LMA=0,
    GLOBAL=1,
    BAYES=2

class FitParameters:
    """
    Simple object for controlling the lifetime fitting parameters.
    """
    def __init__(self,
                 noise_model:None | NoiseModel = None,
                 algorithm: Algorithm=None,
                 intensity_threshold: int=0,
                 intensity_threshold_percent: int=0,
                 lifetime_range: tuple = None,
                 fit_range: tuple = None,
                 num_lifetime_components: int = 1):
        """
        Constructor for FitParameters.

        :param noise_model: the noise model that we will use for fitting. Default is "Poisson Fit".
        :param algorithm: the algorithm that will be used for fitting. Default is LMA (Marquardt-Levenberg Algorithm).
        :param intensity_threshold:
        :param lifetime_range: the range of fluorescence lifetimes to consider. Calculated values outside this range
        are set to zero. Units seconds.
        :param fit_range: The indices of the time-bins to start and end the fitting.
        :param num_lifetime_components: the number of lifetime components to fit.
        :param intensity_threshold: intensity threshold value (overrides percentage).
        :param intensity_threshold_percent: intensity threshold percentage.
        """
        self.noise_model = noise_model if noise_model is not None else NoiseModel.POISSON_FIT
        self.algorithm = algorithm if algorithm is not None else Algorithm.LMA
        self.intensity_threshold = intensity_threshold
        self.intensity_threshold_percent = intensity_threshold_percent
        self.lifetime_range = lifetime_range

        # Check that the fit range is valid
        assert fit_range[1] > fit_range[0], "Fit end point must be higher than fit start point!"

        self.fit_range = fit_range

        assert num_lifetime_components <= 3, "Currently we are only able to fit up to 3 lifetime components."
        self.num_lifetime_components = num_lifetime_components


class LifetimeCalculator:
    """
    Object to handle all lifetime fitting. This object should be initialised once and then used via its class methods.
    """
    def __init__(self, imagej_path: str, log: bool=True):
        """
        Constructor for LifetimeCalculator.

        :param imagej_path: This should be the path to a local Fiji install (Note: it must be Fiji, and you must have both
        Flimlib and FlimOps installed via the update manager)
        :param log: whether to output log information.
        """
        self._image_j_path = imagej_path
        self._log = log

        try:
            # Attempt to create the bridge to imagej from python.
            # Note: there can be significant overhead when bridging, so this should be initialised only once. The class
            # is designed to be used independently of the data source - just re-use the class methods for lifetime
            # fitting.
            logger.info("Attempting to create bridge to imagej...")

            # The line below attempts to find the Fiji install and creates a headless instance.
            # If the path is to a valid installation of ImageJ, then this line should work correctly.
            self._ij = self.__create_imagej_bridge(imagej_path=imagej_path)

            # The next lines require the compiled java classes for both FlimJ and FlimOps to be present in the Fiji
            # installation. These can be installed from the updater within Fiji (use the 'manage update sites' feature).
            self._flimj_noise_types = imagej.sj.jimport('flimlib.NoiseType')
            self._flim_j_fit_functions = imagej.sj.jimport('flimlib.FitFunc')
            self._fit_function = self._flim_j_fit_functions.GCI_MULTIEXP_TAU
            self._flimj_fit_params = imagej.sj.jimport('flimlib.flimj.FitParams')

            # Finally, we create an instance of the flimj parameters. This should only be communicated with from within
            # this class.
            self._fit_parameters = self._flimj_fit_params()

        except Exception as e:
            logger.error("Could not create bridge to imagej. Check java is installed and path is correct.")
            raise e

    # ---------------------------------------------- Private methods ---------------------------------------------------

    def __create_imagej_bridge(self, imagej_path):
        """Create an instance of ImageJ"""
        bridge_start_time = time.time()
        ij = imagej.init(imagej_path)
        if self._log: logger.info(f"Created imagej bridge in {round(time.time() - bridge_start_time, 2)} seconds.")
        return ij

    def _get_flimj_noise_parameter(self, noise_model: NoiseModel):
        """
        Get the corresponding flimlib parameter for a given noise model.
        :param noise_model: the noise model to use
        :return: the corresponding FlimLib parameter.
        """
        value = noise_model.value[0]
        match value:
            case 0:
                return self._flimj_noise_types.NOISE_GAUSSIAN_FIT
            case 1:
                return self._flimj_noise_types.NOISE_POISSON_FIT
            case 2:
                return self._flimj_noise_types.NOISE_POISSON_DATA
            case 3:
                return self._flimj_noise_types.NOISE_MLE

    @staticmethod
    def _get_flimj_algorithm(algorithm: Algorithm):
        """
        Get the string that Flimlib expects for a given fitting algorithm.

        :param algorithm: the algorithm to use for fitting.
        :return: the corresponding string that flimlib can parse.
        """
        value = algorithm.value[0]
        match value:
            case 0:
                return "flim.fitLMA"
            case 1:
                return "flim.fitGlobal"
            case 2:
                raise ValueError("Not yet implemented!")

    # ------------------------------------------ Public methods --------------------------------------------------------
    def calculate_lifetimes(self,
                            tcspc_image: TCSPCImage,
                            fit_parameters: FitParameters=None,
                            use_irf: bool=False,
                            initial_guess: list | None = None,
                            chai_squared_target: float | None = None,
                            ):
        """
        Calculate the fluorescence lifetime for each pixel in a TCSPC image.

        :param tcspc_image: the TPSCImage to calculate lifetimes for.
        :param fit_parameters: object specifying the parameters used for fitting. If unspecified, default values are used.
        :param use_irf: whether to use the IRF when fitting.
        :param initial_guess: an optional ordered initial guess to the fit parameters.
        :param chai_squared_target: the target chai squared value to reach when fitting. Optional.
        :return: the fitted lifetime values
        """
        start_time = time.time()
        if fit_parameters is None:
            fit_parameters = FitParameters()

        # Check that the fit-range is compatible with the image
        assert fit_parameters.fit_range[1] < tcspc_image.shape[2], "Fit end is outside range!"

        # The fitting seems to require the axes in the following order, despite us setting the lifetime axis below.
        # This seems to mirror what is expected in the FlimJ plugin - though I can't see this documented anywhere.
        tcspc_image_ = np.swapaxes(np.swapaxes(tcspc_image.image.copy(), 0, 2), 1, 2)

        # Send the numpy array to the JVM
        self._fit_parameters.transMap = self._ij.py.to_java(tcspc_image_)

        # Set the Flimlib parameters
        self._fit_parameters.ltAxis = 2
        if chai_squared_target is not None: self._fit_parameters.chisq_target = chai_squared_target
        self._fit_parameters.nComp = fit_parameters.num_lifetime_components
        self._fit_parameters.xInc = tcspc_image.t_bin_width
        self._fit_parameters.noise = self._get_flimj_noise_parameter(fit_parameters.noise_model)
        self._fit_parameters.getFittedMap = True
        self._fit_parameters.getResidualsMap = True
        self._fit_parameters.getChisqMap = True
        self._fit_parameters.iThresh = fit_parameters.intensity_threshold # lauren did this
        self._fit_parameters.iThreshPercent = fit_parameters.intensity_threshold_percent

        if initial_guess is not None:
            self._fit_parameters.param = initial_guess

        # Set the fit range if we have specified
        if fit_parameters.fit_range is not None:
            self._fit_parameters.fitStart = fit_parameters.fit_range[0]
            self._fit_parameters.fitEnd = fit_parameters.fit_range[1]

        # Use IRF if requested
        if use_irf:
            if tcspc_image.irf is None:
                logger.warning("Was asked to use IRF but no calibration is available. Continuing without...")
            else:
                self._fit_parameters.instr = tcspc_image.irf

        # Fit the data using FlimJ
        results = self._ij.op().run("flim.fitLMA", self._fit_parameters)

        # Retrieve from the JVM
        fit_results = self._ij.py.from_java(results.paramMap)
        fitted_map = self._ij.py.from_java(results.fittedMap)
        residuals_map = self._ij.py.from_java(results.residualsMap)
        chai_squared_map = self._ij.py.from_java(results.chisqMap)


        lifetime_fit_result = LifetimeFitResult(intensity_image=tcspc_image.intensity_image,
                                                tcspc_image=tcspc_image,
                                                lifetime_fit_result=fit_results,
                                                fitted_map=fitted_map,
                                                residuals_map=residuals_map,
                                                chai_squared_map=chai_squared_map,
                                                pixel_width=tcspc_image.pixel_width,
                                                pixel_height=tcspc_image.pixel_height,
                                                fit_parameters=fit_parameters)

        if self._log: logger.info(f"Calculated lifetime image in {round(time.time() - start_time, 3)} seconds.")

        return lifetime_fit_result


class LifetimeFitResult:
    def __init__(self,
                 intensity_image: np.ndarray,
                 tcspc_image: TCSPCImage,
                 lifetime_fit_result: np.ndarray,
                 fitted_map: np.ndarray,
                 residuals_map: np.ndarray,
                 chai_squared_map: np.ndarray,
                 pixel_width: float,
                 pixel_height: float,
                 fit_parameters: FitParameters):
        """
        Constructor for LifetimeImage.

        :param intensity_image: the associated intensity image.
        :param tcspc_image: the TCSPC image from which lifetimes were calculated.
        :param lifetime_fit_result: the lifetime data. Should be same dimensions as intensity image.
        :param fitted_map: the map of fitted histograms
        :param residuals_map: the map of residuals from the fitted histograms
        :param chai_squared_map: the map of chai squared values from the fitted histograms
        :param pixel_width: the size of the pixels in the image in x. Units microns.
        :param pixel_height: the size of the pixels in the image in y. Units microns.
        :param fit_parameters: the parameters used during fitting.
        """
        # Assign internals
        self._intensity_image = intensity_image
        self._tcspc_image = tcspc_image
        self._lifetime_fit_result = lifetime_fit_result
        self._pixel_width = pixel_width
        self._pixel_height = pixel_height
        self._fit_parameters = fit_parameters
        self._fitted_map = fitted_map
        self._residuals_map = residuals_map
        self._chai_squared_map = chai_squared_map

    # ---------------------------------------------- Private properties ------------------------------------------------
    @property
    def lifetime_fit_result(self):
        """Getter for the raw lifetime image."""
        return self._lifetime_fit_result

    @property
    def pixel_width(self):
        """Getter for the pixel x dimension. Units microns."""
        return self._pixel_width

    @property
    def pixel_height(self):
        """Getter for the pixel y dimension. Units microns."""
        return self._pixel_height

    @property
    def z_image(self):
        """Getter for the z (offset) image. Returns an array of dimension [x,y] with the z value in each pixel."""
        return self._lifetime_fit_result[0, :, :].copy()

    @property
    def a0_image(self):
        """Getter for the A0 (intensity for the first lifetime component) image. Returns an array of dimension [x, y]
        with the A0 value in each pixel."""
        return self._lifetime_fit_result[1, :, :].copy()

    @property
    def t0_image(self):
        """Getter for the t0 (first lifetime) image. Returns an array of dimension [x, y] with the t0 value in each
        pixel."""
        t0_image = self._lifetime_fit_result[2, :, :].copy()
        if self._fit_parameters.lifetime_range is not None:
            t0_image[t0_image <= self._fit_parameters.lifetime_range[0]] = 0
            t0_image[t0_image >= self._fit_parameters.lifetime_range[1]] = 0
        return t0_image

    @property
    def a1_image(self):
        """Getter for the A1 (intensity for the second lifetime component) image. Only defined if the number of
        fit components is >1. Returns an array of dimension [x, y] with the a1 value in each pixel."""
        assert self._fit_parameters.num_lifetime_components > 1, (f"Not defined! We only fitted "
                                                                  f"{self._fit_parameters.num_lifetime_components} "
                                                                  f"lifetime component(s).")

        return self._lifetime_fit_result[3, :, :].copy()

    @property
    def t1_image(self):
        """Getter for the t1 (second lifetime) image. Returns an array of dimension [x, y] with the t1 value in each
        pixel."""
        assert self._fit_parameters.num_lifetime_components > 1, (f"Not defined! We only fitted "
                                                                  f"{self._fit_parameters.num_lifetime_components} "
                                                                  f"lifetime component(s).")
        t1_image = self._lifetime_fit_result[4, :, :].copy()
        if self._fit_parameters.lifetime_range is not None:
            t1_image[t1_image <= self._fit_parameters.lifetime_range[0]] = 0
            t1_image[t1_image >= self._fit_parameters.lifetime_range[1]] = 0

        return t1_image

    @property
    def a2_image(self):
        """Getter fot the A2 (intensity for the third lifetime component) image. OOnly defined if the number of fit
        components >2. Returns an array of dimension [x, y] with the a2 value in each pixel."""

        assert self._fit_parameters.num_lifetime_components > 2, (f"Not defined! We only fitted "
                                                                  f"{self._fit_parameters.num_lifetime_components} "
                                                                  f"lifetime component(s).")

        return self._lifetime_fit_result[5, :, ::].copy()

    @property
    def t2_image(self):
        """Getter for the t2 (third lifetime) image. Returns an array of dimension [x, y] with the t2 value in each
        pixel."""
        assert self._fit_parameters.num_lifetime_components > 2, (f"Not defined! We only fitted "
                                                                  f"{self._fit_parameters.num_lifetime_components} "
                                                                  f"lifetime component(s).")
        t2_image = self._lifetime_fit_result[6, :, :].copy()
        if self._fit_parameters.lifetime_range is not None:
            t2_image[t2_image <= self._fit_parameters.lifetime_range[0]] = 0
            t2_image[t2_image >= self._fit_parameters.lifetime_range[1]] = 0

        return t2_image

    @property
    def average_lifetime_image(self):
        """Getter for the average lifetime image."""
        if self._fit_parameters.num_lifetime_components == 1:
            return self.t0_image
        else:
            a0 = self.a0_image
            a0[a0 <= 0] = 1e-10 # To avoid divide by zero errors

            a1 = self.a1_image
            a1[a1 <= 0] = 1e-10 # To avoid divide by zero errors

            average =  ((a0 * self.t0_image) + (a1* self.t1_image)) / (a0 + a1)
            average[average <= self._fit_parameters.lifetime_range[0]] = 0
            average[average >= self._fit_parameters.lifetime_range[1]] = 0

            return average

    @property
    def fitted_map(self):
        """Getter for the map of fitted histograms."""
        return self._fitted_map

    @property
    def residuals_map(self):
        """Getter for the map of fitted residuals."""
        return self._residuals_map

    @property
    def chai_squared_map(self):
        """Getter for the map of chai squared values."""
        return self._chai_squared_map

    @property
    def tcspc_image(self) -> TCSPCImage:
        """Getter for the TCSPC image"""
        return self._tcspc_image

    # ---------------------------------------------- Private methods ---------------------------------------------------
    def _get_lifetime_image(self, which="t0"):
        """
        Get the specified fluorescence lifetime image.

        :param which: which lifetime image to return. Must be one of ["t0", "t1", or "average."]
        """
        assert which in ["t0", "t1", "average"]

        if which == "t0":
            return self.t0_image

        elif which == "t1":
            return self.t1_image

        else:
            return self.average_lifetime_image

    # ---------------------------------------------- Public methods ----------------------------------------------------
    def colormap_image(self,
                       which: str="t0",
                       cmap: str="turbo",
                       lifetime_max: float=10e-9,
                       scale_by_intensity: bool=False,
                       flip_x: bool=False,
                       flip_y: bool = False,
                       rotate: bool=False):
        """
        Apply a colormap to a lifetime image, scaled between 0 and the lifetime_max. Returns an RGB image.

        :param which: which lifetime component to apply the colormap to. Must be one of "t0", "t1", or "average."
        :param cmap: the name of the matplotlib colormap to apply.
        :param lifetime_max: the maximum lifetime for display (i.e. the highest level in the colormap will correspond
        to this value).
        :param scale_by_intensity: whether to scale the lifetime image by the corresponding intensity values.
        :param flip_x: whether to flip the image in x.
        :param flip_y: whether to flip the image in y.
        :param rotate: whether to rotate the image 90 degrees.
        :return: tge RGB image
        """
        lifetime_image = self._get_lifetime_image(which=which)

        color_mapping = plt.get_cmap(cmap)

        # Next we need to scale the lifetime image to the color map.
        if lifetime_max is None: lifetime_max = np.max(lifetime_image)
        scale_factor = 1 / lifetime_max
        lifetime_image_scaled = lifetime_image * scale_factor

        if np.max(lifetime_image_scaled) > 1:
            # If we're here, we have lifetimes in the image that are greater than the display range specified. We show a
            # warning.
            logger.warning("Saturated! Lifetime image contains lifetimes greater than the display range. Either adjust "
                           "the fit range in the fitting parameters, or increase the maximum lifetime for display.")

        # We convert the lifetime image to RGB
        rgb = color_mapping(lifetime_image_scaled)[:, :, :3]

        # Scale by the intensity values if required
        if scale_by_intensity:
            intensity_image_normalised = (self._intensity_image + 0.0) / np.max(self._intensity_image)
            for z in range(3): rgb[:, :, z] *= intensity_image_normalised

        # Apply the required transformations
        if flip_x: rgb = rgb[:, ::-1, :]
        if flip_y: rgb = rgb[::-1, :, :]
        if rotate: rgb = np.rot90(rgb, axes=(0, 1))

        return rgb

    def display(self,
                which="t0",
                cmap: None | str ="viridis",
                scale_by_intensity: bool = False,
                lifetime_max: None | float = None,
                figsize: tuple = None,
                rotate=False,
                flip_x=False,
                flip_y=False,
                save_path=None,
                close_figure=False,
                title=None):
        """
        Convenience method to display the lifetime image alongside a colormap calibration.

        :param which: which lifetime image to display. Must be one of "t0", "t1", or "average".
        :param cmap: the name of the color map to use. Must be a valid matplotlib colormap.
        :param scale_by_intensity: whether to scale the lifetime values by the intensity recorded in each pixel.
        :param lifetime_max: the maximum lifetime for display.
        :param figsize: the size of the figure.
        :param rotate: whether to rotate the image 90 degrees.
        :param flip_y: whether to flip the image in y.
        :param flip_x: whether to flup the image in x.
        :param save_path: the path to save the figure. If not specified, no image will be saved.
        :param close_figure: whether to close the figure environment. Useful if you want to save but not display the
        image.
        :param title: The title of the figure.
        :return: None
        """
        lifetime_image = self._get_lifetime_image(which=which)

        color_mapping = plt.get_cmap(cmap)
        if lifetime_max is None: lifetime_max = np.max(lifetime_image)

        rgb = self.colormap_image(which=which, cmap=cmap, lifetime_max=lifetime_max,
                                  scale_by_intensity=scale_by_intensity, flip_x=flip_x, flip_y=flip_y, rotate=rotate)
        aspect = self.pixel_width / self.pixel_height if not rotate else self.pixel_height / self.pixel_width

        # We create the plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(rgb, aspect=aspect)
        ax.set_title(title)

        scale_bar = ScaleBar(dx=self.pixel_width if rotate else self.pixel_height,
                             color="white",
                             frameon=False,
                             units="um",
                             location="lower right",
                             pad=1,
                             height_fraction=0.02)
        ax.add_artist(scale_bar)
        ax.set_axis_on()

        fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, lifetime_max * 1e9), cmap=color_mapping.name),
                     ax=ax, orientation='vertical', label='Lifetime (ns)')

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")

        if close_figure:
            plt.close()


    def inspect_pixel(self, x, y):
        """
        Inspect the fitting at the specified pixel. This method generates several plots which aim to provide some
        context about the lifetime fit. Since this isn't a core method, we wrap a utility method.

        :param x: the x location of the pixel to inspect
        :param y: the y location of the pixel to inspect
        """

        # Disable error produced by np.log
        np.seterr(divide='ignore')

        # Calculate the number of components in our fit - we can work this out from the shape of our result
        num_components = (self.lifetime_fit_result.shape[0] - 1) // 2

        aspect = self.pixel_height / self.pixel_width

        # Pull out the individual components for ease
        z = self.lifetime_fit_result[0, y, x]
        a0 = self.lifetime_fit_result[1, y, x]
        t0 = self.lifetime_fit_result[2, y, x]
        a1 = self.lifetime_fit_result[3, y, x] if num_components == 2 else 0
        t1 = self.lifetime_fit_result[4, y, x] if num_components == 2 else 0
        a2 = self.lifetime_fit_result[5, y, x] if num_components == 3 else 0
        t2 = self.lifetime_fit_result[6, y, x] if num_components == 3 else 0

        # Pull out the results that we need
        lifetime_image = self._get_lifetime_image(which="average")
        color_mapping = plt.get_cmap("turbo")
        lifetime_max = np.max(lifetime_image)
        chai_squared_map = self.chai_squared_map[0, :, :]
        residuals = self.residuals_map[:, y, x].copy()
        hist = self.tcspc_image.image[y, x, :].copy()
        t = np.linspace(0, len(hist) * self.tcspc_image.t_bin_width, len(hist))
        regen = self.fitted_map[:, y, x].copy()

        start_crop = 20 if hist.argmax() > 20 else hist.argmax()
        t_cropped = t[hist.argmax() - start_crop // self.tcspc_image.t_binning:]
        hist_cropped = hist[hist.argmax() - start_crop // self.tcspc_image.t_binning:]
        log_hist = np.log(hist_cropped)
        log_regen = np.log(regen)

        hist_bg_removed = hist_cropped - z
        hist_bg_removed[hist_bg_removed <= 0] = 0
        log_hist_bg_removed = np.log(hist_bg_removed)
        log_hist_bg_removed[log_hist_bg_removed <= 0] = 0

        regen_bg_removed = regen - z
        regen_bg_removed[regen_bg_removed <= 0] = 0
        log_regen_bg_removed = np.log(regen_bg_removed)
        log_regen_bg_removed[log_regen_bg_removed <= 0] = 0

        # Work out where our fit start and end points are. (We only know these values in the case that they have been
        # specified, so we use our own logic to determine the fit range. This is easy to do - we simply slide the fit
        # along the raw data until the residuals match those returned by flimlib).
        fit_start, fit_end = find_fit_start_and_end(residuals=residuals, histogram=hist, fit=regen)

        # Print some information about the fit
        title_string = f"Inspecting the lifetime fit at pixel: x={x}; y={y}"
        print(f"\n\n{title_string}\n{"_" * len(title_string)}\n")
        print(f"\n* We were asked to perform a fit with {num_components} components.")

        if num_components == 1:
            print(f"* The fit result gives an exponential decay of : y(t) = {z} + {a0} * exp(-t / {t0})")
        elif num_components == 2:
            print(f"* The fit result gives an exponential decay of : y(t) = {z} + {a0} * exp(-t / {t0}) + {a1} * exp(-t / {t1})")
        elif num_components == 3:
            print(
                f"* The fit result gives an exponential decay of : y(t) = {z} + {a0} * exp(-t / {t0}) + {a1} * "
                f"exp(-t / {t1}) + {a2} * exp(-t / {t2})"
            )

        print(f"* This fit gives a chi-squared value of {chai_squared_map[y, x]}")
        print(f"* It gives a mean residual value of {np.mean(residuals)} with standard deviation {np.std(residuals)}")
        print(f"* Histogram contains {np.sum(hist_cropped)} photon events.")

        if fit_start is not None and fit_end is not None:
            print(f"* {np.sum(hist[fit_start : fit_end])} photons were within the fit range.")

        # Generate the lifetime image
        rgb = self.colormap_image(which="average", cmap="turbo", lifetime_max=lifetime_max, scale_by_intensity=False)

        # First, we plot the images
        fig, ax = plt.subplots(1, 4, figsize=(22, 4.5))
        intensity_image = ndimage.median_filter(self.tcspc_image.intensity_image, 3)
        ax[0].imshow(intensity_image, aspect=aspect, cmap="Greys_r")
        ax[0].set_title("Median-filtered intensity image")
        fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(np.min(intensity_image), np.max(intensity_image)),
                                           cmap="Greys_r"),ax=ax[0], orientation='vertical', label='Counts')

        rectangle = patches.Rectangle((x - 1, y - 1), 2, 2, linewidth=2, edgecolor="yellow")
        ax[0].add_patch(rectangle)


        ax[1].set_title("Average Lifetimes")
        ax[1].imshow(rgb, aspect=aspect)
        fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, lifetime_max * 1e9), cmap=color_mapping.name),
                     ax=ax[1], orientation='vertical', label='Lifetime (ns)')

        rectangle = patches.Rectangle((x - 1, y - 1), 2, 2, linewidth=2, edgecolor="white")
        ax[1].add_patch(rectangle)

        # Next we plot the chai squared image
        ax[2].imshow(chai_squared_map, cmap="viridis", aspect=aspect)
        ax[2].set_title("Chai Squared Map")
        fig.colorbar(mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(np.min(chai_squared_map), np.max(chai_squared_map)),
            cmap="viridis"), ax=ax[2], orientation='vertical', label='Chai-Squared Value')
        rectangle = patches.Rectangle((x - 1, y - 1), 2, 2, linewidth=2, edgecolor="white")
        ax[2].add_patch(rectangle)

        # Next, we show the mean residual map
        residual_spread = np.std(self.residuals_map, 0)
        ax[3].imshow(residual_spread, aspect=aspect, cmap="viridis")
        ax[3].set_title("Map of standard deviation in residuals")
        fig.colorbar(mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(np.min(residual_spread), np.max(residual_spread)),
            cmap="viridis"), ax=ax[3], orientation='vertical', label='Residual std')
        rectangle = patches.Rectangle((x - 1, y - 1), 2, 2, linewidth=2, edgecolor="white")
        ax[3].add_patch(rectangle)

        # Next, we plot the histograms (and logarithmic versions)
        fig, ax = plt.subplots(1, 3, figsize=(21, 4))
        ax[0].plot(t_cropped, hist_cropped, ".", color="green", label="Raw histogram")
        ax[0].plot(t[fit_start : fit_end], regen, color="red", label="Lifetime fit")
        ax[0].grid(alpha=0.5)
        ax[0].set_title("Histograms"), ax[0].set_xlabel("t (s)"), ax[0].set_ylabel("Count")

        ax[1].plot(t_cropped, log_hist, ".", color="green", label="Raw histogram")
        ax[1].plot(t[fit_start: fit_end], log_regen, color="red", label="Lifetime Fit")
        ax[1].grid(alpha=0.5)
        ax[1].set_title("Log Histograms"), ax[1].set_xlabel("t (s)"), ax[1].set_ylabel("Log(Count)")

        ax[2].plot(t_cropped, log_hist_bg_removed, ".", color="green", label="Raw histogram")
        ax[2].plot(t[fit_start: fit_end], log_regen_bg_removed, color="red", label="Lifetime Fit")
        ax[2].grid(alpha=0.5)
        ax[2].set_title("Log Histograms After Offset Removed"), ax[2].set_xlabel("t (s)"), ax[2].set_ylabel("Log(Count)")

        for i in range(3):
            y_max = ax[i].get_ylim()[1]
            if fit_start is not None and fit_end is not None:
                ax[i].vlines(x=[t[fit_start], t[fit_end]],
                             ymin=0, ymax=y_max, linestyle="--", linewidth=1, color="grey", label="Fit limits")
            ax[i].legend()

        # Next we plot the residuals
        fig, ax = plt.subplots(1, 2, figsize=(14, 4))
        ax[0].plot(residuals), ax[0].grid(alpha=0.5)
        ax[0].set_title("Residuals of fit"), ax[0].set_xlabel("Index"), ax[0].set_ylabel("Residual")

        ax[1].hist(residuals), ax[1].grid(alpha=0.5)
        ax[1].set_title("Histogram of Residual Values"), ax[1].set_xlabel("Residual Value"), ax[1].set_ylabel("Count")

        np.seterr(divide='warn')


def find_fit_start_and_end(residuals, histogram, fit):
    """
    Method to determine where the fit started and finished.

    :param residuals: the residuals returned by flimlib
    :param histogram: the raw TCSPC decay
    :param fit: the regenerated decay
    :return: indices of the start and end of the fit, respectively.
    """
    slide_length = len(histogram) - len(fit)

    start, end = None, None

    for i in range(slide_length):
        hist_cropped = histogram[i : i + len(fit)].copy()
        current_residuals = hist_cropped - fit

        if np.abs(np.sum(residuals - current_residuals)) <= 0.01 or np.abs(np.sum(-1 * residuals - current_residuals)) <= 0.01:
            start = i
            end = i + len(fit)

    return start, end