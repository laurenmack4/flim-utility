from ..FlimeraData import FlimeraData
from loguru import logger
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def inspect_pixel(data_cube, xy=None, t_bin=40E-12, buffer=10, fit_range=150):
    if xy is None:
        xy = np.unravel_index(np.argmax(np.sum(data_cube, 0)), np.sum(data_cube, 0).shape)

    histogram = data_cube[:, xy[0], xy[1]].copy()
    t_axis = np.linspace(0, len(histogram) * t_bin, len(histogram))

    log_hist = np.log(histogram + 1e-10)
    log_hist[log_hist <= 0] = 0

    fit_start = np.argmax(log_hist) + buffer
    fit_end = fit_start + fit_range
    fit_region = log_hist[fit_start: fit_end].copy()

    t_cropped = np.linspace(0, t_bin * len(fit_region), len(fit_region))
    fit = np.polyfit(x=t_cropped, y=fit_region, deg=1)
    regen = np.poly1d(fit)(t_cropped)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(t_axis, histogram)
    ax[0].set_title("Histogram")
    ax[0].set_xlabel("t (seconds)")
    ax[0].set_ylabel("Count")

    ax[1].plot(t_axis, log_hist, ".", label="Data")
    ax[1].plot(t_cropped + t_bin * fit_start, regen, label="Linear fit")
    ax[1].set_title("Semi log plot")
    ax[1].set_xlabel("t (seconds)")
    ax[1].set_ylabel("log(count)")

    fig.suptitle(f"Inspection of pixel: x={xy[1]}; y={xy[0]}")


def calculate_lifetime_image(data_cube: np.ndarray,
                             photon_threshold: int | float,
                             buffer: int | float,
                             fit_length: int | float,
                             t_bin: float=40E-12,
                             binning: int=1) -> np.ndarray:
    """
    Calculate the lifetime image from a data cube.

    :param data_cube: Flimera data cube, where axis 0 is the histogram.
    :param photon_threshold: the minimum number of photons we require to calculate a lifetime.
    :param buffer: an offset from the histogram peak (to be used when it's possible we have bleed-through of direct scatter)
    :param fit_length: the number of time bins to perform a linear fit over
    :param t_bin: the elapsed time per bin in the data.
    :param binning: the binning to apply in the time domain
    """
    lifetime_image = np.zeros((data_cube.shape[1], data_cube.shape[2]))

    if binning != 1:
        t_bin *= binning
        fit_length = fit_length // 2

    for row in range(lifetime_image.shape[0]):
        for col in range(lifetime_image.shape[1]):

            hist = data_cube[:, row, col]

            if np.sum(hist) < photon_threshold:
                # If we don't have enough photons, we can't calculate a lifetime.
                lifetime_image[row, col] = 0

            else:
                if binning != 1:
                    hist_binned = np.array([hist[2 * t] + hist[2 * t + 1] for t in range(len(hist) // 2)])
                    hist = hist_binned

                log_hist = np.log(hist + 1e-10)
                log_hist[log_hist <= 0] = 0
                hist_max = np.argmax(log_hist)
                fit_range = log_hist[hist_max + buffer: hist_max + buffer + fit_length]

                t_axis = np.linspace(0, len(fit_range) * t_bin, len(fit_range))
                fit = np.polyfit(x=t_axis, y=fit_range, deg=1)

                lifetime = -1 / fit[0]
                lifetime_image[row, col] = lifetime

    return lifetime_image

class OpticalProperties:
    """
    Simple class representing the optical parameters for the experiment.
    """
    def __init__(self, galvo_scaling: float,
                 galvo_amplitude: float,
                 galvo_frequency: float,
                 magnification: float) -> None:
        """
        Constructor for OpticalProperties class.

        :param galvo_scaling: the drive scaling for the galvo, units mV/micron
        :param galvo_amplitude: the amplitude of the galvo scan for the acquisition
        :param galvo_frequency: the frequency of the galvo drive
        :param magnification: the magnification of the optical system.
        """
        self.galvo_scaling = galvo_scaling
        self.galvo_amplitude = galvo_amplitude
        self.galvo_frequency = galvo_frequency
        self.magnification = magnification


class DSLMFlimera(FlimeraData):
    """
    Class representing the special case of our DSLM-FLIM setup with astigmatic imaging optics.
    """
    def __init__(self, file_path: str,
                 optical_properties: OpticalProperties,
                 log: bool=True):
        """
        Constructor for DSLMFlimera.

        :param file_path: the path to the acquisition file
        :param optical_properties: the optical properties relating to the acquisition
        :param log: whether to output log information.
        """
        super().__init__(file_path, log=log, magnification=optical_properties.magnification)

        self.optical_properties = optical_properties

        # We now need to calculate the number of frames per row in our astigmatic imaging setup. For this we need the
        # scan distance, the scan speed, the row height and the time per row.
        self._scan_distance = self._calculate_scan_distance()

        # Next, we calculate the scan speed
        self._scan_speed = self._calculate_scan_speed()

        # Next, we calculate the image-space row height
        self._row_height = self._calculate_row_height()

        # Next, we calculate the time per row
        self._time_per_row = self._calculate_time_per_row()

        # Next, we calculate the total number of rows scanned
        self._number_rows_scanned = self._calculate_number_rows_scanned()

        # Finally, we can calculate the frames per row
        self._frames_per_row = self._calculate_frames_per_row()

        # Calculate our data cube
        self._data_cube = self._calculate_data_cube()

        # Work out where our triggers were
        self._trigger_rows = self._calculate_trigger_rows()


    # -------------------------------------------- Private properties -------------------------------------------------
    @property
    def data_cube(self) -> np.ndarray:
        """The raw lifetime data cube"""
        return self._data_cube


    # -------------------------------------------- Private methods -----------------------------------------------------
    def _calculate_trigger_rows(self):
        """
        Calculate the rows in which we detected external triggers.
        """
        if len(self.external_triggers) == 0:
            return []

        triggers = np.array(self.external_triggers)
        trigger_frames = []

        for trigger in triggers:
            # frame_found = False
            for i in range(len(self.frame_starts) - 1):
                if trigger in range(self.frame_starts[i], self.frame_starts[i+1]):
                    trigger_frames.append(i)

        trigger_frames = np.array(trigger_frames)
        trigger_rows = trigger_frames / self._frames_per_row
        trigger_rows = [int(np.floor(trigger_rows[i])) for i in range(len(trigger_rows))]
        logger.info(f"Triggers found in rows {trigger_rows}")
        return trigger_rows



    def _calculate_scan_distance(self) -> float:
        """Calculate the total distance scanned in a single galvo scan."""
        scan_distance = self.optical_properties.galvo_amplitude * self.optical_properties.galvo_scaling
        if self.log: logger.info(f"Calculated a scan distance of {np.round(scan_distance, 2)} microns...")
        return scan_distance

    def _calculate_scan_speed(self) -> float:
        """ Calculate the scan speed of the galvo in microns/second. """
        scan_speed = self._scan_distance / (1 / self.optical_properties.galvo_frequency)
        if self.log: logger.info(f"Calculated a scan speed of {np.round(scan_speed, 3)} microns/second...")
        return scan_speed

    def _calculate_row_height(self) -> float:
        """ Calculate the height of each row in image space."""
        row_height = self.pixel_height / self.optical_properties.magnification
        if self.log: logger.info(f"Calculated a row height of {np.round(row_height, 3)}microns")
        return row_height

    def _calculate_time_per_row(self) -> float:
        """Calculate the dwell time per row in seconds."""
        time_per_row = self._row_height / self._scan_speed
        if self.log: logger.info(f"Dwell time per row calculated as {np.round(time_per_row, 3)}")
        return time_per_row

    def _calculate_number_rows_scanned(self) -> float:
        """Calculate the total number of rows scanned during the acquisition"""
        total_rows = self.acquisition_length / self._time_per_row
        if self.log: logger.info(f"We scanned {np.round(total_rows, 3)} rows")
        return total_rows

    def _calculate_frames_per_row(self) -> float:
        """Calculate the total number of frames scanned per row"""
        frames_per_row = self.num_frames / self._number_rows_scanned
        if self.log: logger.info(f"We have {np.round(frames_per_row, 3)} frames per row")
        return frames_per_row

    def _calculate_data_cube(self) -> np.ndarray:
        """
        Calculate the raw lifetime data cube for the dataset.
        """
        # Calculate our time bin range
        micro_times_cropped = self.micro_times.copy()[self.markers != 15]

        min_time_bin, max_time_bin = np.min(micro_times_cropped), np.max(micro_times_cropped)
        num_rows = int(np.floor(self._number_rows_scanned))

        data_cube = np.zeros((max_time_bin - min_time_bin + 1, num_rows, self.sensor_width), dtype=np.uint16)

        for row in tqdm(range(num_rows), desc="Calculating data cube"):
            start = self._frame_starts[int(np.floor(self._frames_per_row * row))]
            end = self._frame_starts[int(np.floor(self._frames_per_row * (row + 1)))]

            for event in range(start, end):
                if self.markers[event] == 15:
                    # This is a frame marker, we ignore
                    pass
                else:
                    data_cube[self.micro_times[event] - min_time_bin, row, self.x_positions[event]] += 1

        return data_cube

    # ------------------------------------------ Public  methods ------------------------------------------------------
    def calculate_intensity_image(self, plot: bool=True, display_triggers: bool=True) -> np.ndarray:
        """
        Calculate the intensity image for the acquisition.

        :param plot: whether to display the triggers. Default True.
        :param display_triggers: Whether to overlay the trigger locations. Default true.
        """
        intensity_image = np.sum(self.data_cube, 0)

        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            intensity_imshow = ax.imshow(intensity_image)
            ax.set_xticks([]), ax.set_yticks([])
            plt.colorbar(intensity_imshow, ax=ax, label="Photons")
            ax.set_title("Intensity Image")
            self._add_scalebar(ax_obj=ax)

            if display_triggers:
                ax.hlines(np.array(self._trigger_rows) - 1, xmin=0, xmax=intensity_image.shape[1] - 1, color="red")

        return intensity_image

    def unravel_data_cube(self) -> np.ndarray | None:
        """
        Unravel the data cube, spatially overlaying the data from multiple galvo passes.
        Note: we assume a sawtooth profile and a rising edge trigger signal.
        :return: the unravelled data cube.
        """
        # If we don't have any triggers we can't do this.
        if len(self.external_triggers) == 0:
            if self.log: logger.error("No trigger information contained in stream. Can't continue.")
            return

        # Check if we received the correct number of triggers
        expected_triggers = int(self.acquisition_length * self.optical_properties.galvo_frequency)
        if self.log: logger.info(f"Expecting {expected_triggers} triggers, found {len(self._trigger_rows)}")
        if expected_triggers != len(self._trigger_rows):
            if self.log: logger.error("Recorded triggers don't match the expected triggers.")
            raise ValueError(f"We dropped {np.abs(expected_triggers - len(self._trigger_rows))} triggers. Can't continue.")

        height = self._trigger_rows[1] - self._trigger_rows[0] - 1
        summed = np.zeros((self.data_cube.shape[0], height, self.data_cube.shape[2]), dtype=np.uint16)

        for trigger_index in range(0, len(self._trigger_rows) - 1):
            summed += self.data_cube[:, self._trigger_rows[trigger_index]: self._trigger_rows[trigger_index] + height, :]
            if self.log: logger.info(f"Extracted galvo cycle between triggers {trigger_index} and {trigger_index + 1}")

        # Now we need to process the last trigger
        last_ = self.data_cube.shape[1] - self._trigger_rows[-1]
        # summed[:, :last_, :] += self.data_cube[:, self._trigger_rows[-1]:, :]
        # summed[:, last_:, :] += self.data_cube[:, :self._trigger_rows[0] + 1, :]
        if self.log: logger.info(f"Extracted frame-wrapped galvo cycle")

        # Because of the sawtooth profile, we know where the triggers are positioned in the cycle. We roll the
        # data cube such that the inversion point is in the middle.
        logger.info("Rolling unravelled data cube")
        summed = np.roll(summed, int(summed.shape[1] * 3 / 4), axis=1)

        # Finally, we need to separate the two "passes" from the triangle wave, and combine.
        # If the data cube has an even y dimension, this is easy:
        if summed.shape[1] % 2 == 0:
            top = summed[:, :summed.shape[1] // 2, :]
            bottom_flipped = summed[:, summed.shape[1] // 2:, :][:, ::-1, :]

        else:
            # If we're here, our data cube has an odd dimension in y.
            half_way_point = int(np.floor(summed.shape[1] / 2))
            top = summed[:, :half_way_point + 1, :]
            bottom_flipped = summed[:, half_way_point - 1:, :][:, ::-1, :]

        return top + bottom_flipped[:, :top.shape[1], :]