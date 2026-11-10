import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter

from FlimeraData import FlimeraData, TCSPCImage, NoiseModel, FitParameters, Algorithm, LifetimeCalculator
from FlimeraData.Util.TCSPCImage import TCSPCImage

dim_x = 192
dim_y = 127
width = 18.4 * 1e-6
height = 9.2 * 1e-6


# select method
# method_marker = 1
# method_frames = 0

#flimera_data = FlimeraData(file_path=fname, magnification=40)

def motion_correction_frames(data: FlimeraData, 
                             lifetime_calculator,
                             velocity: float = 0, 
                             num_groups: int = 1,
                             int_thresh_pc: int = 0,
                             bins: int = 1,
                             save: bool = False,
                             ):
    """
    corrects for stage motion by splitting the data into groups of frames and combining their lifetime images.
    Params
    :param data: FlimeraData dataset.
    :param velocity: velocity of stage in metres per second.
    Output
    :return: LifetimeImage dataset with motion correction.
    """

    summed_tcspc_array = None   # empty array to collect frames into

    frames_per_group = data.num_frames // num_groups
    frame_rate = data.frame_rate    # frame rate of flimera
    micro_times = data.micro_times
    t_min = micro_times.min()
    micro_times -= t_min
    t_max = micro_times.max()
    num_bins = t_max + 1

    plt.figure(figsize=(12, 5))
    plt.tight_layout()

    for i in range(num_groups):
        # setting the start and end
        start_frame = i * frames_per_group
        end_frame = (i + 1) * frames_per_group if i < num_groups - 1 else data.num_frames -1
        # making tcspc image between those frames
        group_tcspc_image = data.generate_TCSPC_image(frame_range=[start_frame, end_frame], n_t_bins=num_bins, remove_outliers_t=True)
    
        group_tcspc_image = group_tcspc_image.bin_image(bins, bins, 1)
        # fit range should be automated but currently just selecting visually
        # potential method below
        fit_start = np.median(np.argmax(group_tcspc_image.image, axis=2)).astype(int) + 10
        fit_end = num_bins - 5
        fit_parameters = FitParameters(noise_model=NoiseModel.POISSON_DATA, algorithm=Algorithm.GLOBAL,
                                       intensity_threshold_percent=int_thresh_pc, lifetime_range=(0, 10e-9), 
                                       fit_range=(fit_start, fit_end),num_lifetime_components=1)
        # calculating lifetimes
        group_lifetime_image = lifetime_calculator.calculate_lifetimes(tcspc_image=group_tcspc_image, 
                                                                       fit_parameters=fit_parameters)

        # don't want to plot every single group
        if i % 2 == 0:
            ax = plt.subplot(2, num_groups // 2, i // 2 + 1)
            # display the mean lifetime for each group
            im = ax.imshow(group_lifetime_image.average_lifetime_image*1e9, cmap='turbo')
            cb = plt.colorbar(im, ax=ax, location='bottom', fraction=0.07, pad=0,
                            label='mean t, ns')
            ax.set_axis_off()
            ax.set_title(f'frames {start_frame}:{end_frame}')
        else:
            print(f'{i/num_groups*100} \\% done') # sanity check
        
        group_tcspc_array = group_tcspc_image.image     # actual histogram data
        print(f'Group {i}, frames {start_frame}:{end_frame}, shape {group_tcspc_array.shape}') # sanity check
        
        # shifting them in space
        # frame / (frame/sec / m/sec * m/pix) = pix
        shift = end_frame / ((frame_rate / velocity) * data.pixel_width * 1e-6 / data.magnification)
        shifted_array = np.roll(group_tcspc_array, (shift + 30)/bins, axis=0)   # 1 for x axis, 0 for y
        # methods for adding this array to the total array
        if summed_tcspc_array is None:
            summed_tcspc_array = shifted_array.copy()
        else:
            summed_tcspc_array += shifted_array

    summed_tcspc = TCSPCImage(image=summed_tcspc_array, irf=None, t_bin_width=data.time_bin_width, 
                              pixel_height=data.pixel_height/data.magnification, pixel_width=data.pixel_width/data.magnification)
    summed_lifetime_image = lifetime_calculator.calculate_lifetimes(tcspc_image=summed_tcspc, 
                                                                    fit_parameters=fit_parameters)
    plt.show()

    fig, ax = plt.subplots()
    im = ax.imshow(summed_lifetime_image.average_lifetime_image*1e9, cmap='turbo', aspect=(data.pixel_width/data.pixel_height))
    cb = plt.colorbar(im, ax=ax, location='right', fraction=0.04, pad=0,
                    label='ns')
    ax.set_axis_off()
    ax.set_title(f'Lifetime corrected for velocity = {velocity*-1e6:.3g} micron/s')
    plt.tight_layout()
    if save:
        plt.savefig(f'_lifetime_corrected_FM')
    else:
        plt.show()

    return summed_lifetime_image


def motion_correction_markers(data: FlimeraData, 
                              lifetime_calculator: LifetimeCalculator,
                              velocity: float = 0,
                              bins: int = 1,
                              int_thresh_pc: int = 0,
                              output: bool = True,
                              save: bool = False):
    """
    corrects for stage motion using the x positions of photon events and frame markers
    Params
    :param data: FlimeraData dataset
    :param velocity: velocity of stage in metres per second
    :param bins: spatial binning factor of TCSPC image
    :param int_thresh_pc: percentage of intensity to set threshold for fitting
    :output: whether to present the graphs and images
    :save: whether to save the graphs and images
    Output
    :return: LifetimeImage dataset with motion correction
    """

    m_width = data.pixel_width / data.magnification

    ### hot pixel removal ###
    raw_int = data.generate_intensity_image(remove_outliers=False)
    hot_pixel_map = np.zeros_like(raw_int)
    # changed kernel size from dans' code from 3 to 5
    hot_pixel_map[raw_int > median_filter(raw_int, 5) * 1.5] = 1

    # get the original markers
    x_markers = data.x_positions.copy()
    y_markers = data.y_positions.copy()
    markers = data.markers.copy()

    # mask of photon events
    photon_mask = (markers == 2)
    assert len(photon_mask) == len(markers), \
        f'{np.shape(photon_mask)} != {np.shape(markers)}'

    x_positions = x_markers[photon_mask]
    y_positions = y_markers[photon_mask]

    # mask of hot pixel events
    hot_photon_mask = np.zeros_like(markers, dtype=bool)
    hot_photon_mask[photon_mask] = (hot_pixel_map[y_positions, x_positions] == 1)
    assert len(hot_photon_mask) == len(markers), \
        f'{np.shape(hot_photon_mask)} != {np.shape(markers)}'

    # getting rid of hot pixel events
    markers[hot_photon_mask] = 3
    assert len(np.unique(markers)) == 3, \
        f'missing a marker type, {np.unique(markers)}'

    # now select only good photon events
    photon_events_mask = (markers == 2)    

    ### normal loading from here ###
    micro_times = data.micro_times[photon_events_mask]  # bins
    x_positions = x_markers[photon_events_mask]  # pix
    y_positions = y_markers[photon_events_mask]  # pix

    # setting time
    t_min = micro_times.min()
    micro_times -= t_min
    t_max = micro_times.max()
    num_bins = int(t_max) + 1

    # finding approx macro time from the photon's frame number (ie seconds since frame 0)
    photon_indices = np.flatnonzero(photon_mask)    # indices of nonzero events
    frame_indices = np.searchsorted(data.frame_starts, photon_indices) - 1   # indices of nonzero events wrt frame starts

    photon_macro_times = frame_indices / data.frame_rate    # conversion frame -> seconds
    photon_distances = (photon_macro_times * velocity) / m_width  # conversion metres -> pixels

    # raw position - moving distance + centering correction
    corrected_y_positions = np.rint(y_positions + photon_distances - np.mean(photon_distances)).astype(np.intp, copy=False)

    # validity checker because it messes up bincount if things outside image dims
    # prob fixable somehow without cutting out data
    valid = ((corrected_y_positions >= 0) & (corrected_y_positions < dim_y) & 
             (x_positions >= 0) & (x_positions < dim_x) & 
             (micro_times >= 0) & (micro_times < num_bins))

    # this is straight from the dans code
    lin = (corrected_y_positions[valid] * (dim_x * num_bins) + x_positions[valid] * num_bins + micro_times[valid])

    # actually making the tcspc image
    tcspc_arr = np.bincount(lin, minlength=dim_x * dim_y * num_bins).reshape((dim_y, dim_x, num_bins))

    # in a construct for fitting
    tcspc_imag = TCSPCImage(image=tcspc_arr, irf=None, 
                            t_bin_width=data.time_bin_width, pixel_height=data.pixel_height/data.magnification, 
                            pixel_width=data.pixel_width/data.magnification)
    tcspc_imag = tcspc_imag.bin_image(bins, bins, 1)

    fit_start = np.median(np.argmax(tcspc_imag.image, axis=2)).astype(int) + 100
    fit_end = num_bins - 1
    
    # fitting in flimj
    fit_parameters = FitParameters(noise_model=NoiseModel.POISSON_DATA, algorithm=Algorithm.GLOBAL, 
                                   intensity_threshold_percent=int_thresh_pc, lifetime_range=(0, 8e-9), 
                                   fit_range=(fit_start, fit_end),num_lifetime_components=1)
    lifetime_imag = lifetime_calculator.calculate_lifetimes(tcspc_image=tcspc_imag, 
                                                            fit_parameters=fit_parameters)

    # lets have a look then
    if output:
        fig, ax = plt.subplots(1, 2)
        im = ax[0].imshow(lifetime_imag.average_lifetime_image*1e9, cmap='turbo', aspect=data.pixel_width/data.pixel_height)
        cb = plt.colorbar(im, ax=ax[0], location='right', fraction=0.065, pad=0,
                        label='mean lifetime (ns)')
        ax[0].set_axis_off()
        ax[0].set_title(f'Mean Lifetime')

        im = ax[1].imshow(lifetime_imag.tcspc_image.intensity_image, cmap='gist_gray', aspect=data.pixel_width/data.pixel_height)
        cb = plt.colorbar(im, ax=ax[1], location='right', fraction=0.065, pad=0,
                        label='counts')
        ax[1].set_axis_off()
        ax[1].set_title(f'Intensity')
        plt.suptitle(f'Corrected for velocity = {velocity * -1e6:.3g} micron/s')
        plt.tight_layout()
        if save:
            plt.savefig(f'_lifetime_corrected.jpg')
        else:
            plt.show()

    return lifetime_imag

def estimate_velocity_frames(data: FlimeraData,
                             binning: int = 1):
    """
    estimates the velocity of a moving sample by comparing the location of the brightest (binned)
    pixel at the first 1000 and last 1000 frames.
    Params
    :param data: FlimeraData dataset
    Output
    :return: velocity of the object in metres per second
    """
    # getting the start and end frames
    n_frame = 2000
    begin_frames = data.generate_TCSPC_image(frame_range=[0, n_frame], remove_outliers_t=True)
    end_frames = data.generate_TCSPC_image(frame_range=[data.num_frames-n_frame-1, data.num_frames-1], remove_outliers_t=True)

    # binning to make the brightest area actually the brightest in case any outliers are left
    begin_binned = begin_frames.bin_image(binning, binning, 1)
    end_binned = end_frames.bin_image(binning, binning, 1)

    # want a visual confirmation that it's finding the right locations
    plt.imshow(begin_binned.intensity_image)
    plt.colorbar()
    plt.plot(begin_binned.intensity_argmax[1], begin_binned.intensity_argmax[0], marker='x', color='r', markersize=10, label='start argmax')
    plt.plot(end_binned.intensity_argmax[1], end_binned.intensity_argmax[0], marker='x', color='g', markersize=10, label='end argmax')
    plt.legend()
    plt.show()

    # find the centre of the area of largest mean intensity
    # assuming direction of motion in image is top to bottom
    begin_brightest_pix = begin_binned.intensity_argmax[0]
    end_brightest_pix = end_binned.intensity_argmax[0]

    # compare the locations of the start and end brightest area to get velocity
    distance_moved_pix = end_brightest_pix - begin_brightest_pix
    distance_moved_m = distance_moved_pix * data.pixel_width * 1e-6 * binning / data.magnification
    frames_travelled = data.num_frames - n_frame
    time_travelled = frames_travelled / data.frame_rate
    velocity_estimation = distance_moved_m / time_travelled

    return velocity_estimation


def gauss(x, *p):
    a, mu, sigma, c = p
    return a * np.exp(-(x - mu)**2/(2. * sigma**2)) + c

def estimate_velocity_max(data: FlimeraData,
                          lifetime_calculator,
                          v_estimate: tuple,
                          binning: int = 1
                          ):
    """
    function to estimate the velocity of a moving TCSPC dataset via a maximisation function
    currently: spatial variance of temporal variance
    assumes movement from the right to left of EZTime image view (so +ve to -ve y position for CAF setup)
    Params
    :param data: flimera dataset, photon stream 
    :param lifetime_calculator: lifetime calculator object
    :param v_estimate: an estimated lower and upper bound for velocity in microns per second
    :param binning: spatial binning factor used in reconstructions that estimate velocity
    Output
    :return: velocity of object in metres per second
    """   
    # choose a range of velocities
    v_low, v_high = v_estimate
    v_range = np.linspace(v_low, v_high, 10)

    max_method = np.zeros_like(v_range)
    # make a recon for each velocity
    count = 0
    for v in v_range:
        velocity = v * -1e-6
        recon_tcspc = motion_correction_markers(data, lifetime_calculator, velocity=velocity, output=False, bins=binning)
        # this is the factor used to estimate velocity: spatial variance of temporal variance
        var_temp = np.var(recon_tcspc.tcspc_image.image, axis=2)
        var_spatial = np.var(var_temp)
        max_method[count] = var_spatial
        count += 1
        # plt.imshow(recon_tcspc.tcspc_image.intensity_image)
        # plt.title(f'{velocity}')
        # plt.colorbar()
        # plt.show()
        
    plt.plot(v_range, max_method/np.max(max_method), label=f'Sum squared I: x/y binning={binning}')

    # params for fitting to gaussian
    p0 = [0.5, np.mean(v_range), 1., 0.5]
    low_bounds = [0., np.min(v_range), 0., 0.]
    up_bounds = [1., np.max(v_range), 20., 1.]

    # fitting to the normalised curve because the numbers are so big
    # gaussian fit is useful because velocity range can be quite spread and so peak of ssq might not be ideal velocity
    # this is not slowing things down tremendously, it takes about 3 ms
    coeff, _ = curve_fit(gauss, v_range, max_method/np.max(max_method), p0=p0, 
                                maxfev=1000, bounds=[low_bounds, up_bounds])
    # [0] is amplitude scaling, [1] is mean mu (velocity), [2] is standard dev sigma, [3] is offset

    fit = gauss(np.linspace(v_range.min(), v_range.max(), 100), *coeff)

    plt.plot(np.linspace(v_range.min(), v_range.max(), 100), fit, label=f'Fit: mu={coeff[1]:.3g} sigma={coeff[2]:.3g}')
    plt.xlabel('Velocity (micron/second)')
    plt.ylabel('Sum of squared intensities')
    plt.title('Gaussian fit to SSI')
    plt.legend()
    plt.show()

    return coeff[1] * 1e-6
