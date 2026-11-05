"""
Script containing utility functions and classes for the loading, parsing and processing of Flimera data streams.
"""
import copy

import h5py
import numpy as np
import time
import pickle
from loguru import logger
from scipy import stats, ndimage
from bisect import bisect_left

from .Util.Registration import transform_x_coordinates, transform_y_coordinates, AffineTransform
from .Util.TCSPCImage import TCSPCImage, ROI
from .Util.Util import adjust_timings, show_overview, calculate_saturation_map

#--------------------------------------- CONSTANTS --------------------------------------------------------------------
# The x-spacing of pixels on the sensor (along the camera's longer axis). Units microns.
PIXEL_WIDTH = 18.4

# The y-spacing of pixels on the sensor (along the camera's shorter axis). Units microns.
PIXEL_HEIGHT = 9.2

# The dimension of the camera along its x-axis.
DIMENSION_X = 192

# The dimension of the camera along its y-axis.
DIMENSION_Y = 127

# Metadata key for frame rate.
FRAME_RATE_KEY = "FrameRateHz"

# Metadata key for time bin width.
TIME_BIN_WIDTH_KEY = "TimePerBin"

# The upper threshold for pixels that we deem as "bad".
HIGH_PIXEL_THRESHOLD = 1.5

# The lower threshold for pixels that we deem as "bad"
LOW_PIXEL_THRESHOLD = 0

# The default start and end crop ranges for the IRFs
DEFAULT_IRF_CROP_START = 20
DEFAULT_IRF_CROP_END = 50

class FlimeraData:
    """
    Class for processing Flimera data streams from H5 files.
    """

    def __init__(self,
                 file_path: str,
                 magnification=1,
                 log=True,
                 calibration=None,
                 portion: tuple=None):
        """
        Constructor for FlimeraData.

        :param file_path: the path to the file being processed.
        :param magnification: the magnification of the image onto the Flimera sensor.
        :param log: whether to output logging.
        :param calibration: the timing calibration. Either an instance of FlimeraCalibration, or the
        associated calibration array, or the path to a timing calibration that has been saved to disk.
        :param portion: The fractional range of the file to load. E.g. (0, 1) will load the full dataset; (0.75, 1) will
        load the last 25%.
        """
        self.log = log
        self._magnification = magnification
        self._file_path = file_path
        self._portion = portion
        self._calibration = calibration

        # Check that the portion we have specified is valid
        self._check_portion_valid()

        # Parse the calibration, if specified
        self._timing_calibration, self._dead_pixel_map, self._irf = self._parse_calibration()

        # Read the raw FLIM data and the metadata associated with the acquisition
        self._flim_data = self._read_data()
        self._metadata = dict(self._flim_data.attrs)

        # Convert the photon stream to a numpy array and delete the zero entries
        self._photon_stream = self._convert_photon_stream()

        # Parse the photon stream into its constituent components
        self._micro_times, self._y_positions, self._x_positions, self._markers = self._parse_stream()

        # Parse the frame information
        self._frame_starts, self._frame_ends, self._triggers_rising_edge = self._get_frame_information()

        # Get a list of frames with triggers
        self._frames_with_triggers = self._get_frames_with_triggers()

        # Get some additional information about the acquisition
        self._num_frames = len(self.frame_starts)
        self._acquisition_length = self.num_frames / self.frame_rate

        # Get the outlier map. This is simply an array that shows pixels that are hot/cold/dead.
        self._outlier_map = self._calculate_outlier_map()

    # ---------------------------------- Private properties -----------------------------------------------------------
    # These are properties for which no setter is defined, so they cannot be changed after an instance is created.
    @property
    def file_path(self) -> str:
        """The file path for the dataset."""
        return self._file_path

    @property
    def pixel_width(self) -> float:
        """The width of the pixels along the horizontal ()"""
        return PIXEL_WIDTH

    @property
    def pixel_height(self) -> float:
        return PIXEL_HEIGHT

    @property
    def sensor_width(self) -> int:
        """The dimension of the sensor in x. Units pixels."""
        return DIMENSION_X

    @property
    def sensor_height(self) -> int:
        """The dimension of the sensor in y. Units pixels."""
        return DIMENSION_Y

    @property
    def frame_rate(self) -> int:
        """The frame rate of the Flimera."""
        return self.metadata[FRAME_RATE_KEY][0]

    @property
    def time_bin_width(self) -> float:
        """Getter for the time-bin width"""
        return self.metadata[TIME_BIN_WIDTH_KEY][0]

    @property
    def metadata(self) -> dict:
        """The metadata from the H5 file."""
        return self._metadata

    @property
    def photon_stream(self) -> np.ndarray:
        """The photon stream, parsed into a numpy array"""
        return self._photon_stream

    @property
    def micro_times(self) -> np.ndarray:
        """The micro-times (time bin) for each photon event."""
        return self._micro_times

    @property
    def y_positions(self) -> np.ndarray:
        """The y positions for each photon event"""
        return self._y_positions

    @property
    def x_positions(self) -> np.ndarray:
        """The x positions for each photon event."""
        return self._x_positions

    @property
    def markers(self) -> np.ndarray:
        """The markers contained in the photon stream."""
        return self._markers

    @property
    def frame_starts(self) -> np.ndarray:
        """The starts of new frames in the photon stream."""
        return self._frame_starts

    @property
    def frame_ends(self) -> np.ndarray:
        """The ends of the acquisition in the photon stream"""
        return self._frame_ends

    @property
    def external_triggers(self) -> np.ndarray:
        """Any external triggers present in the photon stream."""
        return self._triggers_rising_edge

    @property
    def frames_with_triggers(self) -> list:
        """Getter for the list of frames that have external triggers."""
        return self._frames_with_triggers

    @property
    def num_frames(self) -> int:
        """The number of frames contained in the photon stream."""
        return self._num_frames

    @property
    def acquisition_length(self) -> float | int:
        """The total length of the acquisition. Units seconds."""
        return self._acquisition_length

    @property
    def timing_calibration(self) -> np.ndarray | None:
        """The timing calibration for the Flimera, if a calibration is available"""
        if self._timing_calibration is None:
            raise AttributeError("No calibration was provided!")
        else:
            return self._timing_calibration

    @property
    def dead_pixel_map(self) -> np.ndarray | None:
        """The dead pixel map for the Flimera, if a calibration is available."""
        if self._dead_pixel_map is None:
            raise AttributeError("No calibration was provided!")
        else:
            return self._dead_pixel_map

    @property
    def pixel_outlier_map(self) -> np.ndarray:
        """Getter for the pixel outlier map, determined by comparing an intensity image to a median-filtered version of
        that same image. Any pixels that stand out from the local median are considered outliers."""
        return self._outlier_map

    @property
    def portion(self) -> tuple:
        """The portion of the dataset that we have read."""
        if self._portion is None: return 0, 1
        else: return self._portion

    @property
    def irf(self) -> np.ndarray:
        """Getter for the IRF image, comprising a 1D IRF for every pixel on the sensor."""
        return self._irf

    @property
    def overview(self):
        """A visual overview of the dataset."""
        return show_overview(flimera_data=self)

    # -------------------------------------- Public properties ---------------------------------------------------------
    @property
    def magnification(self):
        """The magnification of the image onto the Flimera"""
        return self._magnification

    @magnification.setter
    def magnification(self, magnification: int | float):
        if self.log: logger.info(f"Changed magnification to {magnification}")
        self._magnification = magnification

    # -------------------------------------- Private methods -----------------------------------------------------------
    def _calculate_outlier_map(self):
        """
        Calculate a map of pixel outliers.

        This serves as a crude estimate to the location of pixels that are warm/hot/cold/dead.
        We calculate it by comparing an intensity image to a median-filtered version of the same image. Any pixels that
        stand out considerably from the local median are considered outliers.
        """
        # We use a seconds worth of frames to calculate this
        end_frame = self.frame_rate if self.num_frames > self.frame_rate else self.num_frames - 1
        image = self.generate_intensity_image(frame_range=(0, end_frame))

        # Next we create a median filtered version of the image that should mask any bad pixels.
        filtered_image = image.copy()
        median_filtered = ndimage.median_filter(image, 3)

        # Initialise an array to store the location of bad pixels
        outlier_map = np.zeros(filtered_image.shape)

        outlier_map[filtered_image > HIGH_PIXEL_THRESHOLD * median_filtered] = 1
        outlier_map[filtered_image < LOW_PIXEL_THRESHOLD * median_filtered] = 1
        return outlier_map

    def _parse_calibration(self):
        """Pull out the timing calibration and dead pixel map from the calibration object."""
        if isinstance(self._calibration, FlimeraCalibration):
            timing_calibration = self._calibration.timing_calibration
            dead_pixel_map = self._calibration.dead_pixel_map
            irf = self._calibration.irf

        elif isinstance(self._calibration, str):
            # If we're here, we load the calibration from disk.
            try:
                if self.log: logger.info("Loading timing calibration from file")
                with open(self._calibration, "rb") as file:
                    calibration = pickle.load(file)
                    timing_calibration = calibration.timing_calibration
                    dead_pixel_map = calibration.dead_pixel_map
                    irf = calibration.irf
                if self.log: logger.info("Successfully loaded timing calibration from file.")
            except:
                raise FileNotFoundError("Could not load the specified timing calibration!")

        elif isinstance(self._calibration, _Calibration):
            # If we're here, the calibration has been passed as a Calibration object
            timing_calibration = self._calibration.timing_calibration
            dead_pixel_map = self._calibration.dead_pixel_map
            irf = self._calibration.irf

        else:
            # If we're here, no calibration has been provided.
            timing_calibration = None
            dead_pixel_map = None
            irf = None

        return timing_calibration, dead_pixel_map, irf

    def _check_portion_valid(self):
        """Make sure that the file portion we have specified is valid"""
        if self._portion is None: return

        if type(self._portion) is not tuple:
            raise ValueError("The file portion could not be interpreted. "
                             "Please specify as a tuple with the start and end points, e.g. (0, 1) for the full range.")

        if self._portion[0] < 0 or self._portion[1] > 1 or self._portion[1] < 0 or self._portion[1] > 1:
            raise ValueError("The file portion could not be interpreted."
                             "File portion indices should be between 0 and 1.")

        if self._portion[1] <= self._portion[0]:
            raise ValueError("The file portion could not be interpreted. "
                             "The second index must be greater than the first.")

    def _read_data(self):
        """
        Read the photon data from disk.

        :return: the parsed photon data.
        """
        # Attempt to read the file from disk
        try:
            start_time = time.time()
            flim_data = h5py.File(self.file_path)["Streams"]["Photon"]
            read_time = time.time() - start_time
            if self.log:
                logger.info(f"File read from disk in {np.round(read_time, 3)} seconds.")

        except FileNotFoundError:
            # If we're here, we have entered an incorrect file path - so we simply propagate the exception.
            raise

        return flim_data

    def _convert_photon_stream(self) -> np.ndarray:
        """
        Convert the photon stream to a numpy array and discard zero-events.

        :return: the photon stream as a numpy array
        """
        start_time = time.time()

        start_frame = int(self.portion[0] * self._flim_data.size)
        end_frame = int(self.portion[1] * self._flim_data.size)

        if self.log: logger.info(f"Reading dataset from {self.portion[0] * 100}% to {self.portion[1] * 100}%")

        stream = self._flim_data[start_frame: end_frame].astype(np.uint32, copy=False)
        photon_stream = stream[stream != 0]

        reduction_factor = (len(stream) - len(photon_stream)) / len(stream) * 100
        convert_time = time.time() - start_time

        if self.log:
            logger.info(f"Reduced data size by {np.round(reduction_factor, 3)}% in {np.round(convert_time, 3)} seconds.")

        return photon_stream

    def _parse_stream(self) -> tuple:
        """
        Parse the photon stream into its constituent information.

        :return: tuple of micro times, y positions, x positions and markers.
        """
        start_time = time.time()

        micro_times = (self.photon_stream >> 16) & 0xFFF
        y_positions = self.photon_stream & 0xFF
        x_positions = (self.photon_stream >> 8) & 0xFF
        markers = (self.photon_stream >> 28) & 0xF

        parse_time = np.round(time.time() - start_time)
        if self.log:
            logger.info(f"Parsed photon stream in {parse_time} seconds.")
        return micro_times, y_positions, x_positions, markers

    def _get_frames_with_triggers(self) -> np.ndarray:
        """Get a list of all frames that contain external triggers."""
        frames_with_triggers = np.searchsorted(self.frame_starts, self.external_triggers, side="left") - 1
        return frames_with_triggers

    def _get_frame_information(self) -> tuple:
        """
        Extract frame information from the photon event stream.

        Specifically, we look for:
            1. Frame starts -> these indicate the start of a new frame in the photon data
            2. Frame ends -> these indicate the end of the acquisition -> there should only be one
            3. External triggers (only rising edge currently).

        :return: tuple of frame starts, frame ends, and triggers.
        """
        start_time = time.time()

        frame_marker_mask = ((self.photon_stream >> 28) & 0xF) == 0xF
        frame_marker_locations = np.nonzero(frame_marker_mask)[0]

        frame_marker_info = self.photon_stream[frame_marker_locations]
        frame_marker_types = (frame_marker_info >> 16) & 0xFFF

        # We create boolean masks for the marker types
        starts_mask = frame_marker_types == 0x001
        ends_mask = frame_marker_types == 0x002
        triggers_mask = frame_marker_types == 0x010

        frame_starts = frame_marker_locations[starts_mask]
        frame_ends = frame_marker_locations[ends_mask]
        triggers_rising_edge = frame_marker_locations[triggers_mask]

        parse_time = time.time() - start_time

        if self.log:
            logger.info(f"Parsed frame information in {np.round(parse_time, 3)} seconds")

        return frame_starts, frame_ends, triggers_rising_edge

    def _remove_outliers(self, image: np.ndarray):
        """
        Remove outliers from an intensity image.

        Values in pixels that are deemed as outliers are replaced with the local median. This is an alternative to the
        suboptimal approach of median-filtering the entire image.

        :param image: the image to remove outliers for.
        """
        filtered_image = image.copy()
        median_filtered = np.array(ndimage.median_filter(image, 3))
        filtered_image[self.pixel_outlier_map == 1] = median_filtered[self.pixel_outlier_map == 1]
        return filtered_image


    # --------------------------------------- Public Methods -----------------------------------------------------------
    def generate_intensity_image(self,
                                 frame_range: tuple = None,
                                 remove_outliers: bool = False) -> np.ndarray:
        """
        Generate an intensity image using photon events during the specified range of frames.

        If no frame range is specified, we default to using the entire range.
        :param frame_range: The range of frames to consider.
        :param remove_outliers: Whether to filter out the intensity values from pixels deemed to be outliers.
        :return: the resulting intensity image.
        """

        # If no range specified, we use the full range
        if frame_range is None: frame_range = (0, self.num_frames - 1)
        x_in_range = self.x_positions[self.frame_starts[frame_range[0]]: self.frame_starts[frame_range[1]]]
        y_in_range = self.y_positions[self.frame_starts[frame_range[0]]: self.frame_starts[frame_range[1]]]
        markers_in_range = self.markers[self.frame_starts[frame_range[0]]: self.frame_starts[frame_range[1]]]

        # Pull out only our photon events
        x_in_range = x_in_range[markers_in_range == 2]
        y_in_range = y_in_range[markers_in_range == 2]

        # Form our intensity image
        intensity_image = np.bincount(y_in_range * DIMENSION_X + x_in_range,
                                      minlength=np.prod((DIMENSION_Y, DIMENSION_X))).reshape((DIMENSION_Y, DIMENSION_X))

        if remove_outliers:
            intensity_image = self._remove_outliers(image=intensity_image)
        return intensity_image

    def generate_TCSPC_image(self,
                             frame_range: tuple = None,
                             use_timing_calibration: bool=False,
                             remove_outliers_t: bool=False,
                             remove_background: bool=False,
                             background: np.ndarray = None,
                             n_t_bins: int | None = None) -> TCSPCImage:
        """
        Generate a TCSPC image using photon events during the specified range of frames.

        If no frame range is specified, we default to using the entire range.
        :param frame_range: The range of frames to consider.
        :param use_timing_calibration: Whether to adjust the timings according to a timing calibration. If set to true,
        a timing calibration must be specified when loading hte dataset.
        :param remove_background: Whether to remove a background from the TCSPC image.
        :param background: The background to remove from the TCSPC image.
        :param n_t_bins: the number of time bins to use.
        :return: the resulting TCSPC image.
        """
        if use_timing_calibration and self.timing_calibration is None:
            raise ValueError("Can't adjust TCSPC timings: no calibration loaded!")

        start_time = time.time()

        # If no range specified, we use the full range
        if frame_range is None: frame_range = (0, self.num_frames - 1)
        start_index = self.frame_starts[frame_range[0]]
        end_index = self.frame_starts[frame_range[1]]

        # Pull out relevant events
        x_in_range = self.x_positions[start_index:end_index]
        y_in_range = self.y_positions[start_index:end_index]
        micro_times_in_range = self.micro_times[start_index:end_index]
        markers_in_range = self.markers[start_index:end_index]

        # Pull out only our photon events
        photon_mask = (markers_in_range == 2)
        x_in_range = x_in_range[photon_mask]
        y_in_range = y_in_range[photon_mask]
        micro_times_in_range = micro_times_in_range[photon_mask]

        # Get micro time range for bincount
        t_min = micro_times_in_range.min()
        micro_times_in_range -= t_min
        t_max = micro_times_in_range.max()

        # We override this range if we have requested a certain number of time bins
        if n_t_bins is not None:
            offset = int(n_t_bins) - (int(t_max) + 1)
            if offset < 0:
                raise ValueError(f"Requested number of time bins is smaller than the data range in the requested frame "
                                 f"range ({t_max} populated time-bins).")
            micro_times_in_range += offset
            micro_time_range = n_t_bins
        else:
            micro_time_range = t_max + 1

        # Linear index for bincount
        lin = y_in_range * (DIMENSION_X * micro_time_range) + x_in_range * micro_time_range + micro_times_in_range
        tcspc_image = np.bincount(lin, minlength=DIMENSION_Y *
                                                 DIMENSION_X *
                                                 micro_time_range).reshape(DIMENSION_Y, DIMENSION_X, micro_time_range)

        if self.log: logger.info(f"Calculated specified TCSPC image in {round(time.time() - start_time, 3)} seconds.")

        if use_timing_calibration:
            tcspc_image = adjust_timings(tcspc_image=tcspc_image, timing_calibration=self.timing_calibration)

        if remove_background:
            tcspc_image -= background
# this is laurens doing
        if remove_outliers_t:
            t_outlier_map = 1 - np.broadcast_to(self.pixel_outlier_map.reshape(DIMENSION_Y, DIMENSION_X, 1), np.shape(tcspc_image))
            tcspc_image = tcspc_image * t_outlier_map

        return TCSPCImage(image=tcspc_image,
                          irf=self.irf,
                          t_bin_width=self.time_bin_width,
                          pixel_width=self.pixel_width / self.magnification,
                          pixel_height=self.pixel_height / self.magnification,
                          calibrated_for_timing_offsets=use_timing_calibration)

    def transform_coordinates(self, affine_transform: AffineTransform, copy_data: bool = True):
        """
        Transform the x and y coordinates of a data stream according to an affine transform.

        :param affine_transform: the affine transform to apply.
        :param copy_data: whether to transform in-place or return a new object. If set to true, a new object will be
                          returned with the transformation applied, with the original object left unchanged. If set to
                          false, the transformation is applied in-place.
        :return: the dataset with a coordinate transform applied if copy_data set to true. Otherwise, we don't return
                 anything.
        """
        dataset = copy.copy(self) if copy_data else self

        # Transform the coordinates
        x_coordinates_transformed = transform_x_coordinates(x=self.x_positions, y=self.y_positions,
                                                            transform=affine_transform)
        y_coordinates_transformed = transform_y_coordinates(x=self.x_positions, y=self.y_positions,
                                                            transform=affine_transform)

        # Update the dataset
        dataset._x_positions = x_coordinates_transformed
        dataset._y_positions = y_coordinates_transformed

        if copy_data:
            return dataset

    def calculate_saturation_map(self, median_filter_kernel_size: int = 0):
        """
        Calculate a saturation map for the dataset. Returned as an array of percentages.

        :param median_filter_kernel_size: the kernel size for median filtering. Set to zero for no filtering.
        :return: the calculated saturation map.
        """
        return calculate_saturation_map(flimera_dataset=self, median_filter_kernel_size=median_filter_kernel_size)

    @staticmethod
    def query_dataset(file_path):
        """
        Method to query the size of a file before we actually parse it.

        :param file_path: the path to file we are querying
        :return:
        """
        try:
            logger.info(f"Attempting to query {file_path}")
            start_time = time.time()
            flim_data = h5py.File(file_path)["Streams"]["Photon"]
        except FileNotFoundError:
            raise

        logger.info(f"Dataset has {flim_data.ndim} dimensions.")
        logger.info(f"Dataset has {flim_data.size} entries.")
        logger.info(f"Dataset entries are of type {flim_data.dtype}.")
        logger.info(f"Dataset is {round(flim_data.nbytes / (1024 ** 3), 3)}Gb.")
        logger.info(f"Finished querying dataset in {round(time.time() - start_time, 3)} seconds.")



class FlimeraCalibration(FlimeraData):
    """
    This class represents the special case where the FlimeraDataset is a calibration dataset.

    This class extends FlimeraData so inherits all properties from the base class, but has extra properties to store
    calibration details.
    """

    def __init__(self, file_path, magnification=1, log=True, portion=None):
        super().__init__(file_path=file_path, magnification=magnification, log=log, portion=portion)

        # To measure the timing offset and saturation maps we first need to calculate the TCSPC image
        tcspc_image = self.generate_TCSPC_image(frame_range=None)

        self._timing_calibration = self._calculate_timing_calibration(tcspc_image=tcspc_image.image)
        self._saturation_map = self._calculate_saturation_map(tcspc_image=tcspc_image.image)
        self._dead_pixel_map = self._calculate_dead_pixel_map()
        self._irf = self._calculate_irf()

    @property
    def calibration(self):
        """Return the calibration as a calibration object"""
        return self._form_calibration()

    @property
    def timing_calibration(self) -> np.ndarray:
        """Getter for the timing calibration."""
        return self._timing_calibration

    @property
    def saturation_map(self) -> np.ndarray:
        """Getter for the saturation map."""
        return self._saturation_map

    @property
    def dead_pixel_map(self) -> np.ndarray:
        """Getter for the dead pixel map."""
        return self._dead_pixel_map

    @property
    def irf(self) -> np.ndarray:
        """Getter for the 1D IRF for every dataset."""
        return self._irf

    def _calculate_dead_pixel_map(self) -> np.ndarray:
        """
        Calculate a map of dead pixels across the sensor.
        :return: 2D dead pixel map that is 1 if the pixel is dead, otherwise zero.
        """
        dead_pixel_map = np.zeros(self.timing_calibration.shape)
        dead_pixel_map[self.timing_calibration == 0] = 1
        return dead_pixel_map

    def _calculate_saturation_map(self, tcspc_image) -> np.ndarray:
        """
        Generates a map of how saturated the camera is.
        :param tcspc_image: The pre-calculated TCSPC image. Should cover the full frame range.
        :return: the 2D saturation map.
        """
        saturation_map = np.zeros((tcspc_image.shape[0], tcspc_image.shape[1]))
        num_frames = self.num_frames
        num_saturated_pixels = 0
        for y in range(saturation_map.shape[0]):
            for x in range(saturation_map.shape[1]):
                saturation = np.sum(tcspc_image[y, x, :] / num_frames)
                saturation_map[y, x] = saturation

        # We send a warning if the data is saturated - because the calibration might be off.
        fraction_saturation = num_saturated_pixels / (saturation_map.shape[0] * saturation_map.shape[1])
        if num_saturated_pixels > 0.05:
            logger.warning(f"Caution! Data is {round(100*fraction_saturation, 2)}% saturated. Calibration may not work well.")

        return saturation_map

    def _calculate_irf(self, crop_start=DEFAULT_IRF_CROP_START, crop_end=DEFAULT_IRF_CROP_END) -> np.ndarray:
        """
        Get a 1D IRF for every pixel in the dataset.

        :param crop_start: the number of time-bins to crop out before the IRF maximum
        :param crop_end: the number of time-bins to crop out after the IRF maximum
        :return: the 1d IRF, normalised.
        """
        # We disable logging here as generating the TCSPC image will cause lots of log messages.
        log_ = self.log
        self.log = False

        # Calculate our TCSPC image and check our target pixel isn't dead
        tcspc_image = self.generate_TCSPC_image(use_timing_calibration=True)

        # We have corrected for timing clock offsets, so we can safely average the IRFs from multiple pixels
        intensity_image = tcspc_image.intensity_image
        irf_image = np.zeros((intensity_image.shape[0], intensity_image.shape[1], crop_start + crop_end))

        # Keep track of the number of dead pixels
        dead_pixels = 0

        irf = None
        for y in range(self.sensor_height):
            for x in range(self.sensor_width):
                try:
                    irf = tcspc_image.image[y,x, :]
                    irf = irf[irf.argmax() - crop_start : irf.argmax() + crop_end]
                    irf /= np.sum(irf)
                    irf_image[y, x, :] = irf
                except ValueError:
                    dead_pixels += 1
                    irf_image[y, x, :] = -1

        # We replace the IRF for all the dead pixels with a sample IRF. In reality this won't be used for any fitting
        # because we keep track of the dead pixels, it's just so we have something that we could convolve with every
        # pixel.
        for y in range(self.sensor_height):
            for x in range(self.sensor_width):
                if irf_image[y, x, 0] == -1: irf_image[y, x, :] = irf

        # Re-enable logging if required.
        self.log = log_

        if self.log: logger.warning(f"Couldn't calculate an IRF for {dead_pixels} pixels.")

        return irf_image

    def save_calibration(self, file_path):
        """
        Serialize the timing calibration using pickle.

        :param file_path: the path to save the file.
        """
        calibration = self._form_calibration()

        with open(file_path, "wb") as file:
            # noinspection PyTypeChecker
            # The above comment is necessary to suppress an erroneous linter warning, see
            # https://stackoverflow.com/questions/79049420/python3-pickle-expected-type-supportswritebytes-got-binaryio-instead
            pickle.dump(calibration, file)

    def _form_calibration(self):
        """
        Create a numpy array holding the calibration data. The first slice is the timing calibration, the second is the
        dead pixel map.
        :return: the calibration array.
        """
        calibration = _Calibration(self.timing_calibration, self.dead_pixel_map, self.irf)
        return calibration

    @staticmethod
    def _calculate_timing_calibration(tcspc_image, threshold=500) -> np.ndarray:
        """
        Create a 2D image that stores pixel timing offsets.

        Each pixel has dedicated timing electronics and therefore the time axis is offset. This method calculates this
        offset for each pixel with reference to the global mean, and stores the calibrated offset values in a 2D image.

        :param threshold: the timing offset threshold to apply. Pixels with a timing offset greater than this number
        (from the sensor mode) are set to zero, since they are most likely unresponsive/dead.
        :return: map of the calibrated timing offsets.
        """
        # Initialise a 2D image to store timing offset calibration values for each pixel
        timing_image = np.zeros((tcspc_image.shape[0], tcspc_image.shape[1]))

        for y in range(timing_image.shape[0]):
            for x in range(timing_image.shape[1]):
                arg_max = np.argmax(tcspc_image[y, x, :])
                timing_image[y, x] = arg_max

        # We subtract the modal value so the data is centered on 0
        mode = stats.mode(timing_image.flatten())[0]
        timing_image -= mode

        # Finally, we apply a threshold to the data. (This removes any values calculated for dead pixels, which will
        # have a very negative value here.)
        timing_image[timing_image <= -1 * threshold] = 0

        return timing_image

class _Calibration:
    """A simple object that stores the raw calibration data.

    This object, unlike FlimeraCalibration or FlimeraDataset, can be serialised using pickle.

    Intended to be initialised only within this module.
    """
    def __init__(self, timing_calibration, dead_pixel_map, irf):
        self.timing_calibration = timing_calibration
        self.dead_pixel_map = dead_pixel_map
        self.irf = irf