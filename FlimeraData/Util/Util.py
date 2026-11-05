import matplotlib.pyplot as plt
import tifffile
import time
import numpy as np
from tqdm import tqdm

from loguru import logger
from scipy import ndimage
from matplotlib_scalebar.scalebar import ScaleBar

def generate_intensity_sequence(flimera_data,
                                exposure_time: float,
                                remove_outliers: bool = False,
                                save_path: str | None = None) -> np.ndarray:
    """
    Generate an intensity sequence with a mock exposure time from the raw Flimera stream.

    :param flimera_data: the data stream from which to generate the sequence.
    :param exposure_time: the mock exposure time. Units ms.
    :param remove_outliers: whether to apply local median filtering to bad pixels.
    :param save_path: optional, default=None. The path to save the sequence. Note saved if left as None.
    :return: the intensity sequence.
    """
    # Convert exposure time to seconds for ease
    exposure_time_seconds = exposure_time / 1000

    # Calculate the number of frames per mock exposure and the number of timepoints
    frames_per_exposure = int(exposure_time_seconds * flimera_data.frame_rate)
    num_time_points = int(np.floor(flimera_data.acquisition_length / exposure_time_seconds))

    # Initialise numpy array
    sequence = np.zeros((num_time_points, flimera_data.sensor_height, flimera_data.sensor_width))

    for t in tqdm(range(num_time_points), desc="generating time points"):
        start_frame = t * frames_per_exposure
        end_frame = (t + 1) * frames_per_exposure

        sequence[t, :, :] = flimera_data.generate_intensity_image(frame_range=(start_frame, end_frame),
                                                                  remove_outliers=remove_outliers)

    if save_path is not None:
        save_start_time = time.time()
        logger.info(f"Saving sequence to {save_path}")
        resolution = ((1 / (flimera_data.pixel_height / flimera_data.magnification)),
                      (1 / (flimera_data.pixel_width / flimera_data.magnification)))
        tifffile.imwrite(save_path,
                         np.uint16(sequence.reshape(num_time_points, 1, 1, sequence.shape[1], sequence.shape[2])),
                         resolution=resolution,
                         imagej=True, metadata={"axes": "TZCYX",
                                                "finterval": 1 / flimera_data.frame_rate,
                                                "vmin": 0,
                                                "vmax": np.max(sequence)}
                         )
        logger.info(f"Finished saving sequence in {np.round(time.time() - save_start_time, 3)} seconds.")

    return sequence

def calculate_photon_flux(flimera_data, frame_range: tuple = None, plot=True, fig_size: tuple=(6, 4)):
    """
    Plot the photon flux arriving at the Flimera.
    :param plot: whether to plot the flux.
    :param flimera_data: the data stream to plot the flux for.
    :param frame_range: optional, default None. The frame range to consider. Defaults to all if none specified.
    :param fig_size: optional, default (6, 4). the size of the figure for plotting.
    :return: list of total photons received each frame.
    """
    flux = []
    start_frame = 0 if frame_range is None else frame_range[0]
    end_frame = flimera_data.num_frames - 1 if frame_range is None else frame_range[1]

    for frame in tqdm(range(start_frame, end_frame), desc="Calculating flux"):
        flux.append(np.sum(flimera_data.generate_intensity_image(frame_range=(frame, frame + 1))))

    t_axis = np.linspace(0, len(flux) * 1 / flimera_data.frame_rate, len(flux))

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        ax.plot(t_axis, flux, ".")
        ax.set_title("Photon Flux")
        ax.set_xlabel("t (seconds)")
        ax.set_ylabel("Flux")

    return flux

def _add_scalebar(flimera_data, ax_obj) -> None:
    """
    Add a scalebar to an image plot.

    :param flimera_data: the dataset which we are plotting from. Required for pixel dimensions.
    :param ax_obj: the axes onto which to add the scalebar.
    :return:
    """
    scalebar = ScaleBar(dx=flimera_data.pixel_width / flimera_data.magnification, units="um", location="lower right", color="white",
                        frameon=False)
    ax_obj.add_artist(scalebar)

def show_saturation(flimera_data, frame_start_index: int, frame_end_index: int, fig_size=(6, 4)) -> None:
    """
    Display a saturation map for a given frame range.

    :param flimera_data: the data stream to calculate the saturation map from.
    :param frame_start_index: The start frame index.
    :param frame_end_index: The end frame index.
    :param fig_size: optional, default (6, 4). The size of the plot.
    :return:
    """
    image = flimera_data.generate_intensity_image(frame_range=(frame_start_index, frame_end_index))

    num_frames = frame_end_index - frame_start_index

    # Normalise the image. Each pixel can record a maximum of 1 photon per frame, so this normalisation yields the
    # percentage normalisation for each pixel.
    image /= num_frames

    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    imshow = ax.imshow(image, cmap="hot", vmin=0, vmax=1)
    plt.colorbar(imshow, ax=ax, label="Saturation")
    ax.set_title(f"Saturation: frames {frame_start_index}:{frame_end_index}")
    ax.set_xticks([])
    ax.set_yticks([])

    _add_scalebar(flimera_data=flimera_data, ax_obj=ax)


def adjust_timings(tcspc_image, timing_calibration):
    """
    Utility method to shift the timings in a TCSPC image according to a timing offset calibration.
    :param tcspc_image: the TCPSC image to adjust
    :param timing_calibration: the pixel timing calibration image
    :return:
    """
    adjusted_tcspc_image = np.zeros(tcspc_image.shape)
    for y in range(tcspc_image.shape[0]):
        for x in range(tcspc_image.shape[1]):
            adjusted_tcspc_image[y, x, :] = np.roll(tcspc_image[y, x, :], -1 * timing_calibration[y, x])
    return adjusted_tcspc_image

def calculate_saturation_map(flimera_dataset, median_filter_kernel_size=3):
    """
    Generate a saturation map for a flimera dataset.

    :param flimera_dataset: the dataset to calculate the saturation map for.
    :param median_filter_kernel_size: the kernel size for median filtering. Set to zero for no filtering.
    :return: the calculated saturation map. Each pixel value is the percentage saturation for that pixel.
    """

    intensity_image = flimera_dataset.generate_intensity_image()

    if median_filter_kernel_size > 0:
        intensity_image = ndimage.median_filter(intensity_image, median_filter_kernel_size)
    num_frames = flimera_dataset.num_frames

    saturation_map = 100 * (intensity_image + 0.) / num_frames
    return saturation_map

def show_overview(flimera_data):
    """
    Utility method to provide an overview of a Flimera dataset. We show:
        1. The intensity image, as well as a median filtered copy
        2. A saturation map, as well as a median-filtered copy
        3. Histograms for the brightest pixel in each quadrant
        4. Log-histograms for the brightest pixel in each quadrant.

    :param flimera_data: the dataset to show an overview of.
    :return: None
    """
    # Show the dataset title
    title_string = f"Overview for dataset {flimera_data.file_path}"
    print(f"{title_string}\n{"_" * len(title_string)}\n")

    # Print some basic information about the dataset
    print(f"* Acquisition length: {flimera_data.acquisition_length} seconds")
    print(f"* Num frames: {flimera_data.num_frames}")

    # Calculate the images that we need
    intensity_image = flimera_data.generate_intensity_image()
    filtered_intensity_image = ndimage.median_filter(intensity_image, 3)
    tcspc_image = flimera_data.generate_TCSPC_image()

    # Display the raw intensity image, and a median-filtered copy
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    normal_imshow = ax[0].imshow(intensity_image, aspect=2, cmap="Greys_r")
    ax[0].set_title("Intensity Image")
    ax[0].set_axis_off()
    fig.colorbar(normal_imshow, ax=ax[0], label="Counts")

    filtered_imshow = ax[1].imshow(ndimage.median_filter(intensity_image, 3), aspect=2, cmap="Greys_r")
    ax[1].set_title("Median Filtered Intensity Image")
    ax[1].set_axis_off()
    fig.colorbar(filtered_imshow, ax=ax[1], label="Counts")

    # Display a saturation map, and one for the median-filtered copy
    saturation_map = calculate_saturation_map(flimera_dataset=flimera_data, median_filter_kernel_size=1)
    filtered_saturation_map = 100 * (filtered_intensity_image + 0.) / flimera_data.num_frames

    vmax_unfiltered = 100 if np.max(saturation_map > 100) else np.max(saturation_map)
    vmax_filtered = 100 if np.max(filtered_saturation_map > 100) else np.max(filtered_saturation_map)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    normal_imshow = ax[0].imshow(saturation_map, aspect=2, vmax=vmax_unfiltered, cmap="turbo")

    ax[0].set_title("Saturation Map")
    ax[0].set_axis_off()
    fig.colorbar(normal_imshow, ax=ax[0], label="Saturation (%)")

    filtered_imshow = ax[1].imshow(filtered_saturation_map, aspect=2, vmax=vmax_filtered, cmap="turbo")
    ax[1].set_title("Saturation map after median filtering")
    ax[1].set_axis_off()
    fig.colorbar(filtered_imshow, ax=ax[1], label="Saturation (%)")

    # Now plot our histograms
    np.seterr(divide = 'ignore')
    image_tl = filtered_intensity_image[:intensity_image.shape[0] // 2, :intensity_image.shape[1] // 2]
    tcspc_tl = tcspc_image.image[:intensity_image.shape[0] // 2, :intensity_image.shape[1] // 2]
    hist_loc_tl = np.unravel_index(np.argmax(image_tl), image_tl.shape)

    image_tr = filtered_intensity_image[:intensity_image.shape[0] // 2, intensity_image.shape[1] // 2:]
    tcspc_tr = tcspc_image.image[:intensity_image.shape[0] // 2, intensity_image.shape[1] // 2:, :]
    hist_loc_tr = np.unravel_index(np.argmax(image_tr), image_tr.shape)

    image_bl = filtered_intensity_image[intensity_image.shape[0] // 2:, :intensity_image.shape[1] // 2]
    tcspc_bl = tcspc_image.image[intensity_image.shape[0] // 2:, :intensity_image.shape[1] // 2]
    hist_loc_bl = np.unravel_index(np.argmax(image_bl), image_bl.shape)

    image_br = filtered_intensity_image[intensity_image.shape[0] // 2:, intensity_image.shape[1] // 2:]
    tcspc_br = tcspc_image.image[intensity_image.shape[0] // 2:, intensity_image.shape[1] // 2:]
    hist_loc_br = np.unravel_index(np.argmax(image_br), image_br.shape)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(tcspc_tl[hist_loc_tl[0], hist_loc_tl[1], :], ".", label="Max pixel, top left quadrant")
    ax.plot(tcspc_tr[hist_loc_tr[0], hist_loc_tr[1], :], ".", label="Max pixel, top right quadrant")
    ax.plot(tcspc_bl[hist_loc_bl[0], hist_loc_bl[1], :], ".", label="Max pixel, bottom left quadrant")
    ax.plot(tcspc_br[hist_loc_br[0], hist_loc_br[1], :], ".", label="Max pixel, bottom right quadrant")
    ax.legend()
    ax.set_ylabel("Counts")
    ax.set_xlabel("Time bin")
    ax.set_title("Histograms for the brightest pixel in each quadrant")

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(np.log(tcspc_tl[hist_loc_tl[0], hist_loc_tl[1], :]), ".", label="Max pixel, top left quadrant")
    ax.plot(np.log(tcspc_tr[hist_loc_tr[0], hist_loc_tr[1], :]), ".", label="Max pixel, top right quadrant")
    ax.plot(np.log(tcspc_bl[hist_loc_bl[0], hist_loc_bl[1], :]), ".", label="Max pixel, bottom left quadrant")
    ax.plot(np.log(tcspc_br[hist_loc_br[0], hist_loc_br[1], :]), ".", label="Max pixel, bottom right quadrant")
    ax.legend()
    ax.set_ylabel("Log(count)")
    ax.set_xlabel("Time bin")
    ax.set_title("Log Histograms for the brightest pixel in each quadrant")
    np.seterr(divide='warn')