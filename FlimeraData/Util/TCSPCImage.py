import numpy as np
import pickle
import matplotlib.pyplot as plt

from loguru import logger

from .LifetimeFitting.FitBounds import SingleExponentialFitBounds, DoubleExponentialFitBounds
from .LifetimeFitting.Fitting import (fit_single_exponential, bi_exponential_decay, FitResult, fit_double_exponential,
                                      fit_double_exponential_clamped_t0, fit_irf_convolved_single_exponential,
                                      single_exponential_decay, irf_convolved_single_exponential_decay,
                                      fit_irf_convolved_double_exponential, irf_convolved_bi_exponential_decay,
                                      fit_irf_convolved_double_exponential_clamped_t0)

# ------------------------------------------------ Constants ----------------------------------------------------------
IRF_CROP = 20
DISPLAY_FONT_SIZE = 15
TICK_LABEL_SIZE = 12
DISPLAY_CROP = 20

class ROI:
    """
    Simple object representing a 2D ROI on a camera sensor.
    """
    def __init__(self, x: int, y: int, width: int, height: int):
        """
        Constructor for ROI class.

        :param x: the start pixel coordinate in x for the ROI.
        :param y: the start pixel coordinate in y for the ROI.
        :param width: the width of the ROI, units pixels.
        :param height: the height of the ROI, units pixels.
        """
        self._x = x
        self._y = y
        self._width = width
        self._height = height

    @property
    def x(self) -> int:
        """Getter for the start x coordinate."""
        return self._x

    @property
    def y(self) -> int:
        """Getter for the start y coordinate."""
        return self._y

    @property
    def x_end(self) -> int:
        """Getter for the end x coordinate."""
        return self.x + self.width

    @property
    def y_end(self) -> int:
        """Getter for the end y coordinate."""
        return self.y + self.height

    @property
    def width(self) -> int:
        """Getter for the ROI width."""
        return self._width

    @property
    def height(self) -> int:
        """Getter for the ROI height."""
        return self._height


class TCSPCImage:
    """
    Object representing a time-correlated single-photon counting image, consisting of the photon data as well as
    associated metadata.
    """
    def __init__(self,
                 image: np.ndarray,
                 irf: np.ndarray | None,
                 t_bin_width: float,
                 pixel_width: float,
                 pixel_height: float,
                 calibrated_for_timing_offsets: bool=False):
        """
        Constructor for TCSPCImage.

        :param image: the TCSPC image data. Should be 3-dimensional, with axis order YXT.
        :param irf: the IRF image, containing a 1D IRF for every pixel on the sensor.
        :param t_bin_width: the bin width in the time-domain. Units seconds.
        :param pixel_width: the width of the pixels in the image, units microns.
        :param pixel_height: the height of the pixels in the image, units microns.
        :param calibrated_for_timing_offsets: whether the TCSPC image has been shifted in the T domain to correct for
        pixel timing offsets.
        """
        # Assign internals
        self._image = image
        self._irf = irf
        self._t_bin_width = t_bin_width
        self._pixel_width = pixel_width
        self._pixel_height = pixel_height
        self._calibrated_for_timing_offsets = calibrated_for_timing_offsets
        self._t_binning = 1
        self._x_binning = 1
        self._y_binning = 1

    # ---------------------------------- Private Properties ------------------------------------------------------------
    @property
    def image(self) -> np.ndarray:
        """Getter for the TCSPC image data."""
        return self._image

    @property
    def irf(self) -> np.ndarray:
        """Getter for the IRF image, containing a 1D IRF for every pixel on the sensor."""
        return self._irf

    @property
    def t_bin_width(self) -> float:
        """Getter for the time-bin width."""
        return self._t_bin_width

    @property
    def pixel_width(self) -> float:
        """Getter for the pixel width."""
        return self._pixel_width

    @property
    def pixel_height(self) -> float:
        """Getter for the pixel height."""
        return self._pixel_height

    @property
    def calibrated_for_timing_offsets(self) -> bool:
        """Getter for whether the timings have been offset according to a calibration."""
        return self._calibrated_for_timing_offsets

    @property
    def intensity_image(self) -> np.ndarray:
        """Getter for the associated intensity image."""
        return np.sum(self.image, 2)

    @property
    def shape(self) -> tuple:
        """The shape of the TCSPC image."""
        return self.image.shape

    @property
    def ndim(self) -> int:
        """The number of dimensions of the TCSPC image."""
        return self.image.ndim

    @property
    def dtype(self) -> np.dtype:
        """The datatype of the TCSPC image."""
        return self.image.dtype

    @property
    def size(self) -> int:
        """The number of elements of the TCSPC image."""
        return self.image.size

    @property
    def argmax(self) -> tuple:
        """The position of the maximum element in the TCSPC image."""
        return np.unravel_index(self.image.argmax(), self.image.shape)

    @property
    def intensity_argmax(self) -> tuple:
        """The xy position of the pixel that recorded the greatest number of photons."""
        return np.unravel_index(self.intensity_image.argmax(), self.intensity_image.shape)

    @property
    def t_binning(self) -> int:
        """The binning of the raw data in the tau domain."""
        return self._t_binning

    @property
    def x_binning(self) -> int:
        """The binning of the raw data in the x dimension."""
        return self._x_binning

    @property
    def y_binning(self) -> int:
        """The binning of the raw data in the y dimension."""
        return self._y_binning

    # ------------------------------------------- "Dunder" Methods -----------------------------------------------------
    def __getitem__(self, index) -> np.ndarray:
        """
        Index into the TCSPC image array.

        Returns the sliced underlying image array (not a TCSPC Image).
        """
        return self.image[index]

# ------------------------------------------- Public Methods -------------------------------------------------------
    def bin_image(self, binning_y: int, binning_x: int, binning_t: int):
        """
        Bin the TCSPC image by the specified amount along each axis.

        :param binning_y: the binning to apply along y
        :param binning_x: the binning to apply along x
        :param binning_t: the binning to apply in the time domain
        :return: the binned TCSPC image, with updated metadata.
        """
        bins_y = self.image.shape[0] // binning_y
        bins_x = self.image.shape[1] // binning_x
        bins_t = self.image.shape[2] // binning_t

        binned_image = self.image[:bins_y * binning_y,
                       :bins_x * binning_x,
                       :bins_t * binning_t].reshape(bins_y, binning_y, bins_x, binning_x, bins_t, binning_t).sum(
            axis=(5, 3, 1))

        # We must also bin the associated IRFs.
        if self.irf is not None:
            irf_bins_y = self.irf.shape[0] // binning_y
            irf_bins_x = self.irf.shape[1] // binning_x
            irf_bins_t = self.irf.shape[2] // binning_t

            irf_binned = self.irf[:irf_bins_y * binning_y,
                       :irf_bins_x * binning_x,
                       :irf_bins_t * binning_t].reshape(irf_bins_y,
                                                        binning_y, irf_bins_x, binning_x, irf_bins_t, binning_t).sum(
                axis=(5, 3, 1))

        else:
            irf_binned = None

        tcspc_image = TCSPCImage(image=binned_image,
                          irf=irf_binned,
                          t_bin_width=self.t_bin_width * binning_t,
                          pixel_width=self.pixel_width * binning_x,
                          pixel_height=self.pixel_height * binning_y)

        tcspc_image._t_binning *= binning_t
        tcspc_image._y_binning *= binning_y
        tcspc_image._x_binning *= binning_x

        return tcspc_image

    def get_roi(self, roi: ROI):
        """
        Get a ROI from the TCSPC image. Returned as a new TCSPCImage object.
        """
        image = self.image[roi.y: roi.y_end, roi.x: roi.x_end, :].copy()
        return TCSPCImage(image=image,
                          irf=self.irf[roi.y: roi.y_end, roi.x: roi.x_end, :].copy(),
                          t_bin_width=self.t_bin_width,
                          pixel_width=self.pixel_width,
                          pixel_height=self.pixel_height)

    def inspect_histogram(self, x: int, y: int, size: int=1, clip_before=None) -> tuple[np.ndarray, np.ndarray]:
        """
        Quickly pull out a histogram and corresponding time axis from the TCPSC image.

        :param x: the pixel location in x
        :param y: the pixel location in y
        :param size: the number of pixels in xy to combine
        :param clip_before:
        :return: tuple of (t, histogram) where t is the time axis and histogram is the measured decay curve.
        """

        region = self.image[y: y + size, x: x + size, :].copy()
        if size > 1:
            histogram = np.sum(np.sum(region, 0), 0)
        else:
            histogram = region[0,0,:]

        argmax = histogram.argmax()
        if clip_before is not None:
            histogram = histogram[argmax - clip_before:]

        t_axis = np.linspace(0, len(histogram) * self.t_bin_width, len(histogram))

        return t_axis, histogram

    def save(self, file_path):
        """
        Pickle the TCSPC image to a file.

        :param file_path: the path to save the file to.
        """
        # Check our file path ends in ".pkl"
        if file_path[-4:] != ".pkl":
            file_path = f"{file_path}.pkl"

        with open(file_path, "wb") as file:
            # noinspection PyTypeChecker
            pickle.dump(self, file)

    def get_irf_roi(self, roi: ROI, pixel_outlier_map: np.ndarray = None) -> np.ndarray:
        """
        Get an IRF corresponding to the specified ROI.

        @param roi: the ROI we are getting the IRF for.
        @param pixel_outlier_map: the map of pixel outliers, if specified should be 0 for good pixels, 1 elsewhere. Optional.
        @return: the IRF for the specified ROI.
        """
        if roi is None: roi = ROI(x=0, y=0, width=self.shape[1], height=self.shape[0])
        irf_image_cropped = self.irf[roi.y : roi.y_end, roi.x : roi.x_end, :].copy()
        if pixel_outlier_map is None: pixel_outlier_map = np.zeros((roi.height, roi.width))

        irf_summed = np.zeros(irf_image_cropped.shape[2])
        for y in range(roi.height):
            for x in range(roi.width):
                if pixel_outlier_map[y, x] != 1:
                    irf_summed += irf_image_cropped[y, x, :]
        irf_summed = (irf_summed + 0.) / np.sum(irf_summed)

        return  irf_summed

    def get_hist(self, roi: ROI = None, pixel_outlier_map: np.ndarray = None, threshold=0) -> np.ndarray:
        """
        Get a 1D histogram by summing photons from a ROI on the sensor.

        @param roi: the ROI we are summing over
        @param pixel_outlier_map: a map of pixels to consider. Optional. Should be zero if pixel ok, 1 otherwise.
        @param threshold: the intensity threshold, below which pixels will be ignored. Optional.
        @return: the summed 1D histogram.
        """

        hist = np.zeros(self.shape[2])
        intensity = self.intensity_image

        if roi is None: roi = ROI(x=0, y=0, width=self.shape[1], height=self.shape[0])
        if pixel_outlier_map is None:
            pixel_outlier_map = np.zeros(intensity.shape)
        else:
            # If the map is specified, we must crop it down to the ROI.
            pixel_outlier_map = pixel_outlier_map[roi.y: roi.y + roi.height, roi.x: roi.x + roi.width].copy()

        for y in range(roi.y, roi.y + roi.height):
            for x in range(roi.x, roi.x + roi.width):
                if pixel_outlier_map[y, x] != 1:
                    if intensity[y, x] > threshold:
                        hist += self.image[y, x, :].copy()

        return hist


    # ------------------------------------------------- Fitting functions ---------------------------------------------
    def fit_roi_global_single_exponential(self,
                                          roi: ROI,
                                          bounds: SingleExponentialFitBounds,
                                          pixel_outlier_map: np.ndarray = None,
                                          threshold: int = 0,
                                          start_buffer: int = 0,
                                          end_buffer: int = 0,
                                          display: bool = False):
        """
        Fit a single exponential decay to a single, global histogram formed by summing photons in a given ROI.

        @param roi: The ROI to sum photon events over.
        @param bounds: The bounds on each parameter.
        @param pixel_outlier_map: Map of pixels to consider/ignore. Optional. Should be 0 for good pixels; 1 for bad.
        @param threshold: the intensity threshold. Pixels that recorded fewer than this number of pixels are ignored. Optional.
        @param start_buffer: the number of time-bins from the histogram peak to start fitting from. Optional. Default 0.
        @param end_buffer: the number of time-bins from the histogram end to finish fitting at. Optional. Default 0.
        @param display: Whether to display the fit result. Optional. Default False.
        @return:
        """

        # Get the histogram for the ROI
        histogram = self.get_hist(roi=roi, pixel_outlier_map=pixel_outlier_map, threshold=threshold)
        histogram_cropped = histogram[histogram.argmax() + start_buffer :].copy()
        if end_buffer > 0: histogram_cropped = histogram_cropped[:-end_buffer]

        # Perform the single exponential fitting
        fit = fit_single_exponential(histogram=histogram_cropped, t_bin_width=self.t_bin_width, bounds=bounds)

        # Display the fit result if required
        if display:
            _display_histogram_fit(histogram=histogram.copy(),
                                   fit=fit,
                                   start_buffer=start_buffer,
                                   end_buffer=end_buffer,
                                   t_bin_width=self.t_bin_width,
                                   irf_convolved=False,
                                   fit_type="single")

        return fit

    def fit_roi_global_double_exponential(self,
                                          roi: ROI,
                                          bounds: DoubleExponentialFitBounds,
                                          pixel_outlier_map: np.ndarray = None,
                                          threshold: int = 0,
                                          start_buffer: int = 0,
                                          end_buffer: int = 0,
                                          display: bool = False):
        """
        Fit a double exponential decay to a single, global histogram formed by summing photons in a given ROI.

        @param roi: The ROI to sum photon events over.
        @param bounds: The bounds on each parameter.
        @param pixel_outlier_map: Map of pixels to consider/ignore. Optional. Should be 0 for good pixels; 1 for bad.
        @param threshold: the intensity threshold. Pixels that recorded fewer than this number of pixels are ignored. Optional.
        @param start_buffer: the number of time-bins from the histogram peak to start fitting from. Optional. Default 0.
        @param end_buffer: the number of time-bins from the histogram end to finish fitting at. Optional. Default 0.
        @param display: Whether to display the fit result. Optional. Default False.
        @return:
        """
        # Get the histogram for the ROI
        histogram = self.get_hist(roi=roi, pixel_outlier_map=pixel_outlier_map, threshold=threshold)
        histogram_cropped = histogram[histogram.argmax() + start_buffer:].copy()
        if end_buffer > 0: histogram_cropped = histogram_cropped[:-end_buffer]

        # Perform the double exponential fitting
        fit = fit_double_exponential(histogram=histogram_cropped, t_bin_width=self.t_bin_width, bounds=bounds)

        # Display the fit result if required
        if display:
            _display_histogram_fit(histogram=histogram.copy(),
                                   fit=fit,
                                   start_buffer=start_buffer, end_buffer=end_buffer,
                                   t_bin_width=self.t_bin_width, irf_convolved=False, fit_type="double")

        return fit

    def fit_roi_global_double_exponential_fixed_t0(self,
                                                  roi: ROI,
                                                  bounds: DoubleExponentialFitBounds,
                                                  t0: float,
                                                  pixel_outlier_map: np.ndarray = None,
                                                  threshold: int = 0,
                                                  start_buffer: int = 0,
                                                  end_buffer: int = 0,
                                                  display: bool = False):
        """
        Fit a double exponential decay to a single, global histogram formed by summing photons in a given ROI, where
        the first (t0) lifetime component is known - and specified.

        @param roi: The ROI to sum photon events over.
        @param bounds: The bounds on each parameter.
        @param t0: the known t0 value. Units seconds.
        @param pixel_outlier_map: Map of pixels to consider/ignore. Optional. Should be 0 for good pixels; 1 for bad.
        @param threshold: the intensity threshold. Pixels that recorded fewer than this number of pixels are ignored. Optional.
        @param start_buffer: the number of time-bins from the histogram peak to start fitting from. Optional. Default 0.
        @param end_buffer: the number of time-bins from the histogram end to finish fitting at. Optional. Default 0.
        @param display: Whether to display the fit result. Optional. Default False.
        @return:
        """
        logger.info(f"Ignoring bounds on t0 as this value is fixed.")

        # Get the histogram for the ROI
        histogram = self.get_hist(roi=roi, pixel_outlier_map=pixel_outlier_map, threshold=threshold)
        histogram_cropped = histogram[histogram.argmax() + start_buffer:].copy()
        if end_buffer > 0: histogram_cropped = histogram_cropped[:-end_buffer]

        # Perform the double exponential fitting
        fit = fit_double_exponential_clamped_t0(histogram=histogram_cropped,
                                                t_bin_width=self.t_bin_width, bounds=bounds, t0=t0)

        # Display the fit result if required
        if display:
            _display_histogram_fit(histogram=histogram.copy(),
                                   fit=fit,
                                   start_buffer=start_buffer, end_buffer=end_buffer,
                                   t_bin_width=self.t_bin_width, irf_convolved=False, fit_type="double")

        return fit

    def fit_roi_global_single_exponential_irf_convolved(self,
                                          roi: ROI,
                                          bounds: SingleExponentialFitBounds,
                                          pixel_outlier_map: np.ndarray = None,
                                          threshold: int = 0,
                                          start_buffer: int = 0,
                                          end_buffer: int = 0,
                                          display: bool = False):
        """
        Fit a single exponential decay that is convolved with the corresponding IRF to a single, global histogram
        formed by summing photons in a given ROI.

        @param roi: The ROI to sum photon events over.
        @param bounds: The bounds on each parameter.
        @param pixel_outlier_map: Map of pixels to consider/ignore. Optional. Should be 0 for good pixels; 1 for bad.
        @param threshold: the intensity threshold. Pixels that recorded fewer than this number of pixels are ignored. Optional.
        @param start_buffer: the number of time-bins from the histogram peak to start fitting from. Optional. Default 0.
        @param end_buffer: the number of time-bins from the histogram end to finish fitting at. Optional. Default 0.
        @param display: Whether to display the fit result. Optional. Default False.
        @return:
        """
        assert self.irf is not None, "TCSPC Object has no calibration associated! Can't get the IRF."
        if start_buffer != 0: logger.warning(f"Start buffer is set to a non-zero value. The fitting will proceed, but "
                                             f"the fit may not capture the start of the histogram well, because we are "
                                             f"convolving with an IRF.")

        # Get the histogram for the ROI
        histogram = self.get_hist(roi=roi, pixel_outlier_map=pixel_outlier_map, threshold=threshold)
        histogram_cropped = histogram[histogram.argmax() + start_buffer:].copy()
        if end_buffer > 0: histogram_cropped = histogram_cropped[:-end_buffer]

        # Get the appropriate IRF
        irf = self.get_irf_roi(roi=roi, pixel_outlier_map=pixel_outlier_map)

        # Perform the single exponential fitting
        fit = fit_irf_convolved_single_exponential(histogram=histogram_cropped, irf=irf,
                                                   t_bin_width=self.t_bin_width, bounds=bounds)

        # Display the fit result if required
        if display:
            _display_histogram_fit(histogram=histogram.copy(),
                                   fit=fit,
                                   start_buffer=start_buffer, end_buffer=end_buffer, t_bin_width=self.t_bin_width,
                                   irf_convolved=True,
                                   irf=irf,
                                   fit_type="single")

        return fit

    def fit_roi_global_double_exponential_irf_convolved(self,
                                                        roi: ROI,
                                                        bounds: DoubleExponentialFitBounds,
                                                        pixel_outlier_map: np.ndarray = None,
                                                        threshold: int = 0,
                                                        start_buffer: int = 0,
                                                        end_buffer: int = 0,
                                                        display: bool = False):
        """
        Fit a double exponential decay that is convolved with the corresponding IRF to a single, global histogram
        formed by summing photons in a given ROI.

        @param roi: The ROI to sum photon events over.
        @param bounds: The bounds on each parameter.
        @param pixel_outlier_map: Map of pixels to consider/ignore. Optional. Should be 0 for good pixels; 1 for bad.
        @param threshold: the intensity threshold. Pixels that recorded fewer than this number of pixels are ignored. Optional.
        @param start_buffer: the number of time-bins from the histogram peak to start fitting from. Optional. Default 0.
        @param end_buffer: the number of time-bins from the histogram end to finish fitting at. Optional. Default 0.
        @param display: Whether to display the fit result. Optional. Default False.
        @return:
        """
        assert self.irf is not None, "TCSPC Object has no calibration associated! Can't get the IRF."
        if start_buffer != 0: logger.warning(f"Start buffer is set to a non-zero value. The fitting will proceed, but "
                                             f"the fit may not capture the start of the histogram well, because we are "
                                             f"convolving with an IRF.")

        # Get the histogram for the ROI
        histogram = self.get_hist(roi=roi, pixel_outlier_map=pixel_outlier_map, threshold=threshold)
        histogram_cropped = histogram[histogram.argmax() + start_buffer:].copy()
        if end_buffer > 0: histogram_cropped = histogram_cropped[:-end_buffer]

        # Get the appropriate IRF
        irf = self.get_irf_roi(roi=roi, pixel_outlier_map=pixel_outlier_map)

        # Perform the double exponential fitting
        fit = fit_irf_convolved_double_exponential(histogram=histogram_cropped, irf=irf,
                                                   t_bin_width=self.t_bin_width, bounds=bounds)

        # Display the fit result if required
        if display:
            _display_histogram_fit(histogram=histogram.copy(),
                                   fit=fit,
                                   start_buffer=start_buffer, end_buffer=end_buffer, t_bin_width=self.t_bin_width,
                                   irf_convolved=True,
                                   irf=irf,
                                   fit_type="double")

        return fit

    def fit_roi_global_double_exponential_irf_convolved_clamped_t0(self,
                                                                   roi: ROI,
                                                                   t0: float,
                                                                   bounds: DoubleExponentialFitBounds,
                                                                   pixel_outlier_map: np.ndarray = None,
                                                                   threshold: int = 0,
                                                                   start_buffer: int = 0,
                                                                   end_buffer: int = 0,
                                                                   display: bool = False):
        """
        Fit a double exponential decay that is convolved with the corresponding IRF to a single, global histogram
        formed by summing photons in a given ROI, where the first lifetime component is known.

        @param roi: The ROI to sum photon events over.
        @param t0: the known t0 value. Unit seconds.
        @param bounds: The bounds on each parameter.
        @param pixel_outlier_map: Map of pixels to consider/ignore. Optional. Should be 0 for good pixels; 1 for bad.
        @param threshold: the intensity threshold. Pixels that recorded fewer than this number of pixels are ignored. Optional.
        @param start_buffer: the number of time-bins from the histogram peak to start fitting from. Optional. Default 0.
        @param end_buffer: the number of time-bins from the histogram end to finish fitting at. Optional. Default 0.
        @param display: Whether to display the fit result. Optional. Default False.
        @return:
        """
        assert self.irf is not None, "TCSPC Object has no calibration associated! Can't get the IRF."
        if start_buffer != 0: logger.warning(f"Start buffer is set to a non-zero value. The fitting will proceed, but "
                                             f"the fit may not capture the start of the histogram well, because we are "
                                             f"convolving with an IRF.")

        # Get the histogram for the ROI
        histogram = self.get_hist(roi=roi, pixel_outlier_map=pixel_outlier_map, threshold=threshold)
        histogram_cropped = histogram[histogram.argmax() + start_buffer:].copy()
        if end_buffer > 0: histogram_cropped = histogram_cropped[:-end_buffer]

        # Get the appropriate IRF
        irf = self.get_irf_roi(roi=roi, pixel_outlier_map=pixel_outlier_map)

        # Perform the double exponential fitting
        fit = fit_irf_convolved_double_exponential_clamped_t0(histogram=histogram_cropped,
                                                              irf=irf,t0=t0,  t_bin_width=self.t_bin_width, bounds=bounds)

        # Display the fit result if required
        if display:
            _display_histogram_fit(histogram=histogram.copy(),
                                   fit=fit,
                                   start_buffer=start_buffer, end_buffer=end_buffer, t_bin_width=self.t_bin_width,
                                   irf_convolved=True,
                                   irf=irf,
                                   fit_type="double")

        return fit



def _display_histogram_fit(histogram: np.ndarray,
                           fit: FitResult,
                           start_buffer: int, end_buffer: int,
                           t_bin_width: float,
                           irf_convolved: bool = False,
                           irf: np.ndarray=None,
                           fit_type: str =  "single") -> None:
    """
    Display the result of a histogram.

    Creates a 2x1 subplot displaying the fitted curve overlaid onto the raw histogram, on a linear y-axis (left) and
    a logarithmic y-axis (right).

    @param histogram: The histogram that we performed the fit to.
    @param fit: The result of the fit.
    @param start_buffer: The amount by which we buffered the fit at the start.
    @param end_buffer: The amount by which we buffered the fit at the end.
    @param t_bin_width: The width of the time bins.
    """
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))

    # Crop down the full histogram to an appropriate display range
    histogram = histogram[histogram.argmax() - DISPLAY_CROP :]

    # Generate time axes for plotting the full histogram
    t_axis_full = np.linspace(0, len(histogram) * t_bin_width, len(histogram))
    t_axis_cropped = t_axis_full[histogram.argmax() + start_buffer :]
    if end_buffer > 0: t_axis_cropped = t_axis_cropped[:- end_buffer]

    # Regenerate the curve for display
    if fit_type == "double" and irf_convolved == False:
        regen = bi_exponential_decay(t_axis=t_axis_full[:len(t_axis_cropped)], a0=fit.a0, t0=fit.t0, a1=fit.a1, t1=fit.t1, bg=fit.bg)
    elif fit_type == "single" and irf_convolved == False:
        regen = single_exponential_decay(t_axis=t_axis_full[:len(t_axis_cropped)], a0=fit.a0, t0=fit.t0, bg=fit.bg)

    elif fit_type == "single" and irf_convolved == True:
        regen = irf_convolved_single_exponential_decay(time_axis=t_axis_full[:len(t_axis_cropped)], a0=fit.a0, t0=fit.t0, bg=fit.bg, irf=irf)
    elif fit_type == "double" and irf_convolved == True:
        regen = irf_convolved_bi_exponential_decay(time_axis=t_axis_full[:len(t_axis_cropped)], a0=fit.a0, t0=fit.t0, a1=fit.a1, t1=fit.t1, bg=fit.bg, irf=irf)
    else:
        raise ValueError("Invalid fit function specified!")

    ax[0].plot(t_axis_full, histogram, marker=".", markersize=4, linestyle="none", color="green", label="raw")
    ax[0].plot(t_axis_cropped, regen, color="red", label="fit")
    ax[0].legend(fontsize=TICK_LABEL_SIZE)
    ax[0].set_title("Global Histogram", fontsize=DISPLAY_FONT_SIZE)
    ax[0].set_xlabel("time (s)", fontsize=DISPLAY_FONT_SIZE)
    ax[0].set_ylabel("Count", fontsize=DISPLAY_FONT_SIZE)
    ax[0].tick_params(axis='both', which='both', labelsize=TICK_LABEL_SIZE)

    ax[1].semilogy(t_axis_full, histogram, marker=".", markersize=4, linestyle="none", color="green", label="raw")
    ax[1].semilogy(t_axis_cropped, regen, color="red", label="fit")
    ax[1].legend(fontsize=TICK_LABEL_SIZE)
    ax[1].set_title("Global Histogram", fontsize=DISPLAY_FONT_SIZE)
    ax[1].set_xlabel("time (s)", fontsize=DISPLAY_FONT_SIZE)
    ax[1].set_ylabel("Count", fontsize=DISPLAY_FONT_SIZE)
    ax[1].tick_params(axis='both', which='both', labelsize=TICK_LABEL_SIZE)

    # Lines for fit start and end
    ax[0].set_ylim([0, np.max(histogram) * 1.05])
    ax[0].vlines(x=(histogram.argmax() + start_buffer) * t_bin_width, ymin=np.min(histogram), ymax=np.max(histogram) * 2, linestyles="--", color="blue")
    ax[0].vlines(x=(len(histogram) - end_buffer) * t_bin_width, ymin=np.min(histogram), ymax=np.max(histogram) * 2, linestyles="--",
                 color="blue")
    ax[0].axvspan(xmin=(histogram.argmax() + start_buffer) * t_bin_width, xmax=(len(histogram) - end_buffer) * t_bin_width,
                ymin=0, ymax=1, color="blue", alpha=0.15)

    ax[1].vlines(x=(histogram.argmax() + start_buffer) * t_bin_width, ymin=np.min(histogram), ymax=np.max(histogram), linestyles="--",
                 color="blue")
    ax[1].vlines(x=(len(histogram) - end_buffer) * t_bin_width, ymin=np.min(histogram), ymax=np.max(histogram), linestyles="--",
                 color="blue")

    ax[1].set_ylim([np.min(histogram), np.max(histogram) * 1.2])
    ax[1].axvspan(xmin=(histogram.argmax() + start_buffer) * t_bin_width,
                  xmax=(len(histogram) - end_buffer) * t_bin_width,
                  ymin=0, ymax=1, color="blue", alpha=0.15)


def load_tcspc_image(file_path):
    """
    Load a pickled TCSPC image from a pickle.

    :param file_path: the path of the file.
    """

    # Check our file path ends in ".pkl"
    if file_path[-4:] != ".pkl":
        file_path = f"{file_path}.pkl"

    with open(file_path, "rb") as file:
        tcspc_image = pickle.load(file)

    return tcspc_image




