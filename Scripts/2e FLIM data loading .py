  # -*- coding: utf-8 -*-

#%% Imports
import h5py as hp # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt #type: ignore
import cv2 # type: ignore
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit # type: ignore
from phasorpy.plot import plot_phasor, PhasorPlot
from phasorpy.phasor import phasor_center, phasor_from_signal
from phasorpy.component import phasor_component_fraction
from phasorpy.datasets import fetch
from phasorpy.lifetime import (
    phasor_calibrate,
    phasor_from_lifetime,
    phasor_semicircle_intersect,
    phasor_to_lifetime_search,
    phasor_to_normal_lifetime,
)
from scipy.stats import mode
from scipy.ndimage import median_filter
from tifffile import imwrite # type: ignore
from scipy import stats

plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 15,
    "ytick.labelsize": 15,
    "xtick.labelsize": 15,
})

# Functions
def phasor_fit(data, omega, t_data, intens, threshold):
    """ returns the g and s values of a phasor fit """
    g = np.zeros((data.shape[1],data.shape[2]))
    s = np.zeros((data.shape[1],data.shape[2]))
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            if intens[i, j]>threshold:
                g[i, j] = np.sum(data[:,i , j]*np.cos(omega*t_data))/intens[i, j]
                s[i, j] = np.sum(data[:,i , j]*np.sin(omega*t_data))/intens[i, j] # reason for difference between data and intensity is to be able to apply thresholding
            else:
                g[i, j] = np.nan
                s[i, j] = np.nan
    return g,s        

def enhance_local_contrast(img,ks):
    img = np.array(img)
    image_blur = cv2.GaussianBlur(img,(ks,ks),20,20)
    new_image = img - 0.7*image_blur
    new_image = new_image
    return new_image

def com(data, t_data):
    intensity= np.sum(data,axis=0)
    taumap = np.zeros_like(intensity)
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            if intensity[i,j]!=0:
                taumap[i,j] = np.dot(data[:,i,j],t_data)/intensity[i,j]
    return taumap

def exponential_decay(t, A, tau, offset):
    return A * np.exp(-t / tau) + offset

def double_decay(t, A1, A2, tau1, tau2, offset):
    return (A1 * np.exp(-t / tau1)) + (A2 * np.exp(-t / tau2)) + offset

def triple_decay(t, A1, A2, A3, tau1, tau2, tau3, offset):
    return (A1 * np.exp(-t / tau1)) + (A2 * np.exp(-t / tau2)) + (A3 * np.exp(-t / tau3)) + offset

def fit_exponential_decay(data, t_data, intens, threshold=0):
    """
    Fit exponential decay for each pixel in the data array.
    
    Parameters:
    -----------
    data : ndarray (time_bins, height, width)
        SPAD histogram data
    t_data : ndarray (time_bins,)
        Time values corresponding to time bins
    intens : ndarray (height, width)
        Intensity map (sum of photon counts per pixel)
    threshold : float
        Minimum intensity threshold for fitting
    
    Returns:
    --------
    tau_map : ndarray (height, width)
        Lifetime (tau) for each pixel
    A_map : ndarray (height, width)
        Amplitude for each pixel
    offset_map : ndarray (height, width)
        Offset for each pixel
    """
    tau_map = np.zeros((data.shape[1], data.shape[2]))
    A_map = np.zeros((data.shape[1], data.shape[2]))
    offset_map = np.zeros((data.shape[1], data.shape[2]))
    chi_map = np.zeros((data.shape[1], data.shape[2]))
    ''' make zero array for each map '''
    
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            if intens[i, j] > threshold:# threshold at selection of pixels based on intensity
                pixel_decay = data[:, i, j].astype(float)
                
                try:
                    # Initial parameter guesses
                    A_guess = np.max(pixel_decay)
                    tau_guess = t_data[np.argmax(pixel_decay)] / 2
                    offset_guess = np.min(pixel_decay)
                    
                    
                    # Fit the exponential decay
                    popt, _ = curve_fit(
                        exponential_decay, 
                        t_data, 
                        pixel_decay,
                        p0=[A_guess, tau_guess, offset_guess],
                        maxfev=5000
                    )

                    fitted = exponential_decay(t_data, popt[0], popt[1], popt[2])
                    chi = np.sum(((pixel_decay - fitted)**2 / (fitted))) / (len(pixel_decay)-3)
                    
                    A_map[i, j] = popt[0]
                    tau_map[i, j] = popt[1]
                    offset_map[i, j] = popt[2]
                    chi_map[i, j] = chi
                    
                except RuntimeError:
                    # If fitting fails, set to NaN
                    A_map[i, j] = np.nan
                    tau_map[i, j] = np.nan
                    offset_map[i, j] = np.nan
                    chi_map[i, j] =  np.nan
            else:
                A_map[i, j] = np.nan
                tau_map[i, j] = np.nan
                offset_map[i, j] = np.nan
                chi_map[i,j] = np.nan
    
    return tau_map, A_map, chi_map, offset_map

def fit_double_decay(data, t_data, intens, threshold=0):
    """
    Fit double exponential decay for each pixel in the data array.
    
    Parameters:
    -----------
    data : ndarray (time_bins, height, width)
        SPAD histogram data
    t_data : ndarray (time_bins,)
        Time values corresponding to time bins
    intens : ndarray (height, width)
        Intensity map (sum of photon counts per pixel)
    threshold : float
        Minimum intensity threshold for fitting
    
    Returns:
    --------
    tau_map : ndarray (height, width)
        Lifetime (tau) for each pixel
    A_map : ndarray (height, width)
        Amplitude for each pixel
    offset_map : ndarray (height, width)
        Offset for each pixel
    """
    tau1_map = np.zeros((data.shape[1], data.shape[2]))
    A1_map = np.zeros((data.shape[1], data.shape[2]))
    tau2_map = np.zeros((data.shape[1], data.shape[2]))
    A2_map = np.zeros((data.shape[1], data.shape[2]))
    offset_map = np.zeros((data.shape[1], data.shape[2]))
    chi_map = np.zeros((data.shape[1], data.shape[2]))
    ''' make zero array for each map '''

    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            if intens[i, j] > threshold:# threshold at selection of pixels based on intensity
                #pixel_decay = binned[:, i, j].astype(float)
                pixel_decay = data[:, i, j].astype(float)
                
                try:
                    # Initial parameter guesses
                    A1_guess = A2_guess = (np.max(pixel_decay))
                    tau1_guess = 0.1
                    tau2_guess = 2.2
                    offset_guess = 1
                    lower_bounds = [0, 0, 0, 0, 0]
                    upper_bounds = [2*np.max(pixel_decay)+10, 2*np.max(pixel_decay)+10, 20, 20, 0.25*np.max(pixel_decay)+1]
                    
                    # Fit the exponential decay
                    popt, popc = curve_fit(
                        double_decay, 
                        t_data, 
                        pixel_decay,
                        p0=[A1_guess, A2_guess, tau1_guess, tau2_guess, offset_guess],
                        bounds=(lower_bounds, upper_bounds),
                        maxfev=10000
                    )


                    fitted = double_decay(t_data, popt[0], popt[1], popt[2], popt[3], popt[4])
                    chi = np.sum(((pixel_decay - fitted)**2 / (fitted))) / (len(pixel_decay)-5)
                    
                    sums_a = (popt[0] + popt[1])/9
                    A1_map[i, j] = (popt[0]/9)/sums_a
                    A2_map[i, j] = (popt[1]/9)/sums_a
                    tau1_map[i, j] = popt[2]
                    tau2_map[i, j] = popt[3]
                    offset_map[i, j] = popt[4]/9
                    chi_map[i, j] = chi/9

                    
                except RuntimeError:
                    # If fitting fails, set to NaN
                    A1_map[i, j] = np.nan
                    A2_map[i, j] = np.nan
                    tau1_map[i, j] = np.nan
                    tau2_map[i, j] = np.nan
                    offset_map[i, j] = np.nan
                    chi_map[i, j] = np.nan
            else:
                A1_map[i, j] = np.nan
                A2_map[i, j] = np.nan
                tau1_map[i, j] = np.nan
                tau2_map[i, j] = np.nan
                offset_map[i, j] = np.nan
                chi_map[i, j] = np.nan
        print(i/data.shape[1]*100)
    
    return tau1_map, tau2_map, A1_map, A2_map, offset_map, chi_map

def fit_triple_decay(data, t_data, intens, threshold=0, upthreshold=100):
    """
    Fit double exponential decay for each pixel in the data array.
    
    Parameters:
    -----------
    data : ndarray (time_bins, height, width)
        SPAD histogram data
    t_data : ndarray (time_bins,)
        Time values corresponding to time bins
    intens : ndarray (height, width)
        Intensity map (sum of photon counts per pixel)
    threshold : float
        Minimum intensity threshold for fitting
    
    Returns:
    --------
    tau_map : ndarray (height, width)
        Lifetime (tau) for each pixel
    A_map : ndarray (height, width)
        Amplitude for each pixel
    offset_map : ndarray (height, width)
        Offset for each pixel
    """
    tau1_map = np.zeros((data.shape[1], data.shape[2]))
    A1_map = np.zeros((data.shape[1], data.shape[2]))
    tau2_map = np.zeros((data.shape[1], data.shape[2]))
    A2_map = np.zeros((data.shape[1], data.shape[2]))
    tau3_map = np.zeros((data.shape[1], data.shape[2]))
    A3_map = np.zeros((data.shape[1], data.shape[2]))
    offset_map = np.zeros((data.shape[1], data.shape[2]))
    chi_map = np.zeros((data.shape[1], data.shape[2]))
    ''' make zero array for each map '''

    data_padded = np.pad(data, ((0, 0), (1, 1), (1, 1)), mode='constant')
    binned = np.zeros_like(data, dtype=np.result_type(data, np.int64))

    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            binned[:, i, j] = data_padded[:, i:i+3, j:j+3].sum(axis=(1,2))

    
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            if intens[i, j] > threshold and intens[i, j] < upthreshold:# threshold at selection of pixels based on intensity
                #pixel_decay = binned[:, i, j].astype(float)
                pixel_decay = data[:, i, j].astype(float)
                
                try:
                    # Initial parameter guesses
                    A1_guess = A2_guess = A3_guess = (np.max(pixel_decay))
                    # tau1_guess = t_data[np.argmax(pixel_decay)] / 2
                    # tau2_guess = tau1_guess / 1.5
                    # offset_guess = np.min(pixel_decay)
                    # tau1_guess = (time[np.argmax(Pix)] / 2) + 1
                    # tau2_guess = (time[np.argmax(Pix)] / 2) + 1
                    tau1_guess = 0.2
                    tau2_guess = 2
                    tau3_guess = 4
                    offset_guess = 1
                    lower_bounds = [0, 0, 0, 1e-3, 1e-3, 1e-3, 0]
                    upper_bounds = [np.max(pixel_decay)+10, np.max(pixel_decay)+10, np.max(pixel_decay)+10, 10, 10, 10, 10]
                    #print(A1_guess)
                    
                    # Fit the exponential decay
                    popt, popc = curve_fit(
                        triple_decay, 
                        t_data, 
                        pixel_decay,
                        p0=[A1_guess, A2_guess, A3_guess, tau1_guess, tau2_guess, tau3_guess, offset_guess],
                        bounds=(lower_bounds, upper_bounds),
                        maxfev=10000
                    )

                    #unc = np.sqrt(np.max(pixel_decay))**2
                    unc = np.var(pixel_decay, ddof=7)
                    fitted = triple_decay(t_data, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
                    chi = np.sum(((pixel_decay - fitted)**2 / (fitted))) / (len(pixel_decay)-7)
                    #chi = chisquare(pixel_decay, fitted*(np.sum(pixel_decay)/np.sum(fitted)), ddof=5)[0] / (len(fitted)-5)
                    
                    sums_a = (popt[0] + popt[1] + popt[2])/9
                    A1_map[i, j] = (popt[0]/9)/sums_a
                    A2_map[i, j] = (popt[1]/9)/sums_a
                    A3_map[i, j] = (popt[2]/9)/sums_a
                    tau1_map[i, j] = popt[3]
                    tau2_map[i, j] = popt[4]
                    tau3_map[i, j] = popt[5]
                    offset_map[i, j] = popt[6]
                    chi_map[i, j] = chi/9


                    
                except RuntimeError:
                    # If fitting fails, set to NaN
                    A1_map[i, j] = np.nan
                    A2_map[i, j] = np.nan
                    A3_map[i, j] = np.nan
                    tau1_map[i, j] = np.nan
                    tau2_map[i, j] = np.nan
                    tau3_map[i, j] = np.nan
                    offset_map[i, j] = np.nan
                    chi_map[i, j] = np.nan
            else:
                A1_map[i, j] = np.nan
                A2_map[i, j] = np.nan
                A3_map[i, j] = np.nan
                tau1_map[i, j] = np.nan
                tau2_map[i, j] = np.nan
                tau3_map[i, j] = np.nan
                offset_map[i, j] = np.nan
                chi_map[i, j] = np.nan
    
    return tau1_map, tau2_map, tau3_map, A1_map, A2_map, A3_map, offset_map, chi_map

def calculate_timing_calibration(tcspc_image, threshold=500):
    """
    Create a 2D image that stores pixel timing offsets.

    Each pixel has dedicated timing electronics and therefore the time axis is offset. This method calculates this
    offset for each pixel with reference to the global mean, and stores the calibrated offset values in a 2D image.

    :param threshold: the timing offset threshold to apply. Pixels with a timing offset greater than this number
    (from the sensor mode) are set to zero, since they are most likely unresponsive/dead.
    :return: map of the calibrated timing offsets.
    """
    # Initialise a 2D image to store timing offset calibration values for each pixel
    timing_image = np.zeros((tcspc_image.shape[1], tcspc_image.shape[2]))

    for y in range(timing_image.shape[0]):
        for x in range(timing_image.shape[1]):
            arg_max = np.argmax(tcspc_image[:, y, x])
            timing_image[y, x] = arg_max

    # We subtract the modal value so the data is centered on 0
    #modev = mode(timing_image.flatten(), axis=None, keepdims=True)[0]
    #modev = mode(timing_image.flatten())
    modev = mode(timing_image.flatten())[0]
    print(modev)
    timing_image -= modev

    # Finally, we apply a threshold to the data. (This removes any values calculated for dead pixels, which will
    # have a very negative value here.)
    timing_image[timing_image <= -1 * threshold] = 0

    return timing_image

def adjust_timings(tcspc_image, timing_calibration):
    """
    Utility method to shift the timings in a TCSPC image according to a timing offset calibration.
    :param tcspc_image: the TCPSC image to adjust
    :param timing_calibration: the pixel timing calibration image
    :return:
    """
    adjusted_tcspc_image = np.zeros(tcspc_image.shape)
    for y in range(tcspc_image.shape[1]):
        for x in range(tcspc_image.shape[2]):
            adjusted_tcspc_image[:, y, x] = np.roll(tcspc_image[:, y, x], (-1 * timing_calibration[y, x]))
    return adjusted_tcspc_image

def calculate_outlier_map(image, high, low):
    """
    Calculate a map of pixel outliers.

    This serves as a crude estimate to the location of pixels that are warm/hot/cold/dead.
    We calculate it by comparing an intensity image to a median-filtered version of the same image. Any pixels that
    stand out considerably from the local median are considered outliers.
    """

    # Next we create a median filtered version of the image that should mask any bad pixels.
    filtered_image = image.copy()
    median_filtered = median_filter(image, 3)

    # Initialise an array to store the location of bad pixels
    outlier_map = np.zeros(filtered_image.shape)

    outlier_map[filtered_image > high * median_filtered] = 1
    outlier_map[filtered_image < low * median_filtered] = 1
    return outlier_map


def remove_outliers(image: np.ndarray, outlier_map):
    """
    Remove outliers from an intensity image.

    Values in pixels that are deemed as outliers are replaced with the local median. This is an alternative to the
    suboptimal approach of median-filtering the entire image.

    :param image: the image to remove outliers for.
    """
    filtered_image = image.copy()
    median_filtered = np.array(median_filter(image, 3))
    filtered_image[outlier_map == 1] = median_filtered[outlier_map == 1]
    return filtered_image

def residuals(data, fit):
    '''
    Finds the residual between data and fit.
    Parameters:
    :param data: (ndarray): observed data
    :param fit: (ndarray): fit model, same size as data
    Returns:
    :return: array of residuals
    '''
    resarray = np.zeros_like(data)
    for i in np.arange(len(data)):
        resarray[i] = data[i] - fit[i]
    return resarray

color_mapping = plt.get_cmap('turbo')
def intensity_weight_map(map, intensity, name, parent):
    """
    :param map: the lifetime or contribution map to be weighted
    :param intensity: intensity to scale map to
    :param name: name to save image as
    """
    max = np.max(map)
    scale = 1/max
    img_scaled = map * scale
    rgb = color_mapping(img_scaled)[:, :, :3]
    int_norm = intensity / np.max(intensity)
    for z in range(3): rgb[:, :, z] *= int_norm
    fig, ax= plt.subplots(figsize=(7,5))
    im = ax.imshow(rgb, cmap='turbo', vmax=max)
    cb = plt.colorbar(im, ax=ax, location='bottom', fraction=0.062, pad=0.00,
                    label=f'{name} (ns)')
    ax.set_axis_off()
    plt.savefig(f'{parent}{name}_iweighted')
#======================================================
#%% Get files

loc = './all_images/'
file_list = ['c_1_hists.h5', 'c_2_hists.h5', 'c_3_hists.h5', 
             'c_4_hists.h5', 'c_5_hists.h5', 'c_6_hists.h5', 
             'c_7_hists.h5', 'c_8_hists.h5', '10m_1_hists.h5', 
             '10m_2_hists.h5', '10m_4_hists.h5', '10m_5_hists.h5', '1h_1_hists.h5',
             '24h_1_hists.h5', '24h_2_hists.h5', '24h_3_hists.h5', 
             '24h_4_hists.h5', '24h_5_hists.h5', '24h_6_hists.h5', 
             '24h_7_hists.h5']
# file_list = ['mcf_wash_1.h5', 'mcf_wash_2.h5', 'mcf_wash_3.h5', 
#              'mcf_wash_triangle.h5', 'mcf_wash_half.h5']

for i, fname in enumerate(file_list):
    named = fname[:-8]
    path=r"C:\Users\laure\MSc\Project\FLIM-Utility-main\Notebooks\Data\3107" + "/" + named + "hists.h5"
    bg_path = r"C:\Users\laure\MSc\Project\FLIM-Utility-main\Notebooks\Data\3107\10m_bg.h5"
    f=hp.File(path,'r') # Load in file from path
    bg_f = hp.File(bg_path, 'r')
    irf = hp.File(r"C:\Users\laure\MSc\Project\FLIM-Utility-main\Notebooks\Data\3107\irf_nopixoffset_hists.h5")
    dset=f['FLIM']
    bg_dset = bg_f['FLIM']

    # getting dataset keys to obtain time info
    mylist=list(f.keys())
    keys=list(dset.keys())
    attrs=dict(dset.attrs)
    bintime = np.asarray(attrs['TimePerBin']).item()
    nano_bintime = bintime * 1e9

    # printing all the keys in the dataset
    dataset_names=[]
    for name, obj in dset.items():
        if isinstance(obj, hp.Group):
            print("Group:", name)
        elif isinstance(obj, hp.Dataset):
            dataset_names.append(name)
            print("Dataset:", name, "shape=", obj.shape, "dtype=", obj.dtype)



    # load spad histogram from file in pixel map shape

    f_flim=f['FLIM']['Histogram']
    bg_flim = bg_f['FLIM']['Histogram']
    irf_flim = irf['FLIM']['Histogram']

    spad = np.zeros((len(f_flim[0]),len(f_flim))) # create an array with 0s the same shape as histogram data
    bg_spad = np.zeros((len(bg_flim[0]),len(bg_flim)))
    irf_spad = np.zeros((len(irf_flim[0]),len(irf_flim)))


    for i in range(len(f_flim)):
        spad[:,i] = f_flim[i] # for each value in flim allocate to SPAD variable
        bg_spad[:,i] = bg_flim[i]
        irf_spad[:,i] = irf_flim[i]

    irf_spad = irf_spad[:np.shape(spad)[0], :]

    width = int(len(f_flim)/192) # divide length of shape by height for width
    spad = spad.reshape((-1,192,126)) # Reshape spad
    bg_spad = bg_spad.reshape((-1,192,126))
    irf_spad = irf_spad.reshape((-1,192,126))

    spad_og = spad.copy()
    bg_spad_og = bg_spad.copy()
    irf_spad_og = irf_spad.copy()

    # background correction
    spad = spad - bg_spad 
    print(np.min(spad[0]))


    # Plotting one pixel 
    # choosing the two pixels of intersest, one in the corner where no cell (prob)
    x=3
    y=4
    x2 = 100
    y2 = 100

    # if you want pix 1 to be the brightest pixel (usually bad hot pixel):
    #x2, y2 = np.unravel_index(np.sum(spad, axis=0).argmax(), np.sum(spad, axis=0).shape)

    #time=np.arange(0,(nano_bintime*len(spad)),nano_bintime) # legacy time, found that there were issues so changed to linspace
    time = np.linspace(0, nano_bintime*len(spad), num=np.int32((nano_bintime*len(spad))/nano_bintime))
    time_og = time

    Pix2 = spad[:, y2, x2]
    Pix = spad[:, y, x]

    plt.figure(figsize=(8,6))
    plt.scatter(time, Pix, label=(f'Pixel 1 at ({x},{y})'))
    plt.scatter(time, Pix2, label=(f'Pixel 2 at ({x2},{y2})'))
    plt.legend()
    plt.title(f'Pixel decay with background subtraction')
    plt.show()

    # doing 3x3 binning because the signal is weak and fitting is bad

    data_padded = np.pad(spad, ((0, 0), (1, 1), (1, 1)), mode='constant') # padding so image stays same shape
    binned = np.zeros_like(spad, dtype=np.result_type(spad, np.int64))
    for i in range(spad.shape[1]):
        for j in range(spad.shape[2]):
            binned[:, i, j] = data_padded[:, i:i+3, j:j+3].sum(axis=(1,2))


    # technically do not need the background spad but is sometimes useful to image later
    bg_padded = np.pad(bg_spad, ((0, 0), (1, 1), (1, 1)), mode='constant')
    bg_binned = np.zeros_like(bg_spad, dtype=np.result_type(spad, np.int64))
    for i in range(bg_spad.shape[1]):
        for j in range(bg_spad.shape[2]):
            bg_binned[:, i, j] = bg_padded[:, i:i+3, j:j+3].sum(axis=(1,2))


    Pix2_bin = binned[:, y2, x2]
    Pix_bin = binned[:, y, x]


    plt.figure(figsize=(8,6))
    plt.scatter(time, Pix_bin, label=(f'Binned Pixel 1 at ({x},{y})'))
    plt.scatter(time, Pix2_bin, label=(f'Binned Pixel 2 at ({x2},{y2})'))
    plt.legend()
    plt.title(f'Pixel decay with subtraction and 3x3 Binning')
    plt.show()


    # timing calibration 

    cal_im = calculate_timing_calibration(spad)
    spad = adjust_timings(spad, cal_im)

    cal_im_bin = calculate_timing_calibration(binned)
    binned = adjust_timings(binned, cal_im_bin)

    cal_bg = calculate_timing_calibration(bg_spad)
    bg_spad = adjust_timings(bg_spad, cal_bg)

    cal_bg_bin = calculate_timing_calibration(bg_binned)
    bg_spad_bin = adjust_timings(bg_binned, cal_bg_bin)

    # spad has changed so resetting the pixels
    Pix2 = spad[:, y2, x2]
    Pix = spad[:, y, x] 
    Pix2_bin = binned[:, y2, x2]
    Pix_bin = binned[:,y,x]


    # plt.figure(figsize=(8,6))
    # plt.scatter(time, Pix_bin, label=(f'Pixel 1 at {x},{y}'))
    # plt.scatter(time, Pix2_bin, label=(f'Pixel 2 {x2},{y2}'))
    # plt.legend()
    # plt.title(f'Pixel decay (3x3 bin), time calibrated')
    # plt.show()


    spad = np.repeat(spad, 2, axis=-1)
    bg_spad = np.repeat(bg_spad, 2, axis=-1)
    irf_spad = np.repeat(irf_spad, 2, axis=-1)
    binned = np.repeat(binned, 2, axis=-1)
    bg_binned = np.repeat(bg_binned, 2, axis=-1)


    spad_ogx2 = spad # repeat each value beside itself
    bg_spad_ogx2 = bg_spad
    irf_spad_ogx2 = irf_spad
    binned_ogx2 = binned
    bg_binned_ogx2 = bg_binned


    # Tail extraction and checking with pixels

    # a little scuffed but just extracting tail by eye
    tail = 80
    end = -1
    plt.figure(figsize=(8,6))
    plt.scatter(time, Pix_bin, label=(f'Pixel 1 at {x},{y}'))
    plt.scatter(time, Pix2_bin, label=(f'Pixel 2 {x2},{y2}'))
    plt.legend()
    plt.vlines(x=time[tail], ymin=np.min(Pix2_bin), ymax=np.max(Pix2_bin), color='black')
    plt.vlines(x=time[end], ymin=np.min(Pix2_bin), ymax=np.max(Pix2_bin), color='black')

    plt.title(f'Tail extraction')


    # clipping data to exclude end where its rolled backwards

    spad_tail = spad_ogx2[tail:end, :, :]
    nano_tail_time = time[tail:end]
    spad = spad_tail

    bg_spad_tail = bg_spad_ogx2[tail:end, :,:]
    bg_spad = bg_spad_tail

    binned_tail = binned_ogx2[tail:end, :, :]
    binned = binned_tail

    bg_binned_tail = bg_binned_ogx2[tail:end, :, :]
    bg_binned = bg_binned_tail


    Pix = Pix[tail:end]
    Pix2 = Pix2[tail:end]

    Pix2_bin = Pix2_bin[tail:end]
    Pix_bin = Pix_bin[tail:end]

    # cleaning the pixel

    # only want the binned one now
    Pix = Pix_bin
    Pix2 = Pix2_bin


    threshold=np.percentile(Pix, 0) #define threshold
    dead = np.max(Pix.flatten())
    #Pix[Pix<threshold]= np.nan #apply thresholding
    #Pix[Pix>=dead]= np.nan 
    #Pix[Pix<=0]= 0 
    Pix_outliers = calculate_outlier_map(Pix, 1.5, 0.25)
    Pix = remove_outliers(Pix, Pix_outliers)

    threshold2=np.percentile(Pix2, 0) #define threshold
    dead = np.max(Pix2.flatten())
    #Pix2[Pix2<threshold2]= np.nan #apply thresholding
    #Pix2[Pix2>=dead]= np.nan
    #Pix2[Pix2<=0]= 0
    Pix2_outliers = calculate_outlier_map(Pix2, 1.5, 0.25)
    Pix2 = remove_outliers(Pix2, Pix2_outliers)



    #

    # setting initial params
    A1_guess = A2_guess = np.max(Pix2)
    tau1_guess = 0.1 # get the tau lifetimes based on phasor plot in later analysis and come back after
    tau2_guess = 2.2
    offset_guess = 0
    print(A1_guess, tau1_guess)

    # reasonable bounds
    lower_bounds = [0, 0, 0, 0, 0]
    upper_bounds = [2*np.max(Pix2)+1, 2*np.max(Pix2)+1, 20, 20, 1000]


    # fit the double exponential decay to each pix
    popt1,pcov1 = curve_fit(
        double_decay, 
        nano_tail_time, 
        Pix, 
        p0 = [A1_guess, A2_guess, tau1_guess, tau2_guess, offset_guess],
        maxfev=10000,
        bounds=(lower_bounds, upper_bounds),
        nan_policy='omit'
    )


    popt2,pcov2 = curve_fit(
        double_decay, 
        nano_tail_time, 
        Pix2,
        p0 = [A1_guess, A2_guess, tau1_guess, tau2_guess, offset_guess],
        maxfev=10000,
        bounds=(lower_bounds, upper_bounds),
        nan_policy='omit'
    )


    fit=double_decay(nano_tail_time, *popt1 )
    fit2=double_decay(nano_tail_time, *popt2 )

    fig, ax = plt.subplots(3, 1, figsize=(10,10), gridspec_kw={'height_ratios': [4, 1, 1]})

    ax[0].scatter(nano_tail_time, Pix, label=(f'Pixel 1 at {x},{y}'), marker='x')
    ax[0].plot(nano_tail_time, fit, color='b',label=f'Fit 1: A1=%5.3f, A2=%5.3f, tau1=%5.3f, tau2=%5.3f, Offset=%5.3f' % tuple(popt1))
    ax[0].scatter(nano_tail_time, Pix2, label=(f'Pixel 2 at {x2},{y2}'))
    ax[0].plot(nano_tail_time, fit2, color='r',label=f'Fit 2: A1=%5.3f, A2=%5.3f, tau1=%5.3f, tau2=%5.3f, Offset=%5.3f' % tuple(popt2), linestyle='--')
    ax[0].legend()
    ax[0].set_title(f'Tail Fitting')


    # finding chi and residuals...
    chi = (np.sum(((Pix - fit)**2 / (fit))) / (len(Pix)-5)) #/9 to account for 3x3 binning
    chi2 = (np.sum(((Pix2 - fit2)**2 / (fit2))) / (len(Pix2)-5))

    print(f'reduced chi squared pix 1 {chi:5f} and pix 2 {chi2:5f}')

    ax[1].set_title('Residuals')
    ax[1].plot(nano_tail_time,residuals(Pix, fit), alpha=1, label=(f'Pixel 1 at {x},{y}'))
    ax[1].plot(nano_tail_time,residuals(Pix2, fit2), alpha=1, linestyle='--',label=(f'Pixel 2 at {x2},{y2}'))
    ax[1].grid()
    ax[1].legend(loc='lower right')
    ax[2].set_title('Residuals Histogram')
    ax[2].hist(residuals(Pix, fit), alpha=0.5, label=(f'Pixel 1 at {x},{y}'))
    ax[2].hist(residuals(Pix2, fit2), alpha=0.5,label=(f'Pixel 2 at {x2},{y2}'))
    ax[2].legend()
    fig.tight_layout()
    plt.show()


    # Get intensity      
    bg = np.sum(bg_spad, axis=0)
    intens = np.sum(spad,axis=0) # sum photon counts

    threshold=np.percentile(intens, 50) #define threshold
    dead = np.max(intens.flatten())
    #intens[intens<threshold]= 0 #apply thresholding
    intens[intens>=dead] = 0 
    intens[intens<0] = 0 
    intens_outliers = calculate_outlier_map(intens, 1.5, 0.25)
    intens = remove_outliers(intens, intens_outliers)

    #  decay fitting the whole data
    #t_data = np.arange(bintime,bintime*(spad.shape[0]+1),bintime) # again changed this
    # sometimes need to do +1 to the num steps, think there is an odd/even thing going on
    t_data = np.linspace(nano_bintime, nano_bintime*(spad.shape[0]), num=np.int32(spad.shape[0]+0))


    # make sure threshold is set how you want- defines lower limit of intensities to fit
    tau1_map, tau2_map, A1_map, A2_map, offset_map, chi_map= fit_double_decay(binned, 
                                                                            t_data, intens,
                                                                            threshold=np.percentile(intens,40))


    # cleaning up tau and A maps

    tau1_map[np.isnan(tau1_map)] = 0 #set nans to 0
    tau1_map[np.isinf(tau1_map)] = 0 #set infitities to 0
    tau1_map[tau1_map<0] = 0 # set anything below 0 to 0
    tau1_outliers = calculate_outlier_map(tau1_map, 1.5, 0.25)
    tau1_map = remove_outliers(tau1_map, tau1_outliers)

    tau2_map[np.isnan(tau2_map)] = 0 #set nans to 0
    tau2_map[np.isinf(tau2_map)] = 0 #set infitities to 0
    tau2_map[tau2_map<0] = 0 # set anything below 0 to 0
    tau2_outliers = calculate_outlier_map(tau2_map, 1.5, 0.25)
    tau2_map = remove_outliers(tau2_map, tau2_outliers)


    A1_map[np.isnan(A1_map)] = 0 #set nans to 0
    A1_map[np.isinf(A1_map)] = 0 #set infitities to 0
    A1_map[A1_map <= 0] = 1e-12
    A1_outliers = calculate_outlier_map(A1_map, 1.5, 0.25)
    A1_map = remove_outliers(A1_map, A1_outliers)

    A2_map[np.isnan(A2_map)] = 0 #set nans to 0
    A2_map[np.isinf(A2_map)] = 0 #set infitities to 0
    A2_map[A2_map <= 0] = 1e-12
    A2_outliers = calculate_outlier_map(A2_map, 1.5, 0.25)
    A2_map = remove_outliers(A2_map, A2_outliers)


    chi_map[np.isinf(chi_map)] = np.nan #set infitities to 0




    # cleaning up tau and A maps

    tau_percentile = 98
    a_percentile = 98

    tau1_avg = np.mean(tau1_map)
    tau1_max= np.max(tau1_map)
    cleanp1 = np.clip(tau1_map, 45e-3, np.percentile(tau1_map,tau_percentile))
    cleanm1=np.clip(tau1_map,None,(tau1_max-(1/4*tau1_avg)))

    cleana1 =np.clip(A1_map, None, np.percentile(A1_map, a_percentile))

    tau2_avg = np.mean(tau2_map)
    tau2_max= np.max(tau2_map)
    cleanp2 = np.clip(tau2_map, 45e-3, np.percentile(tau2_map, tau_percentile))
    cleanm2=np.clip(tau2_map,None,(tau2_max-(1/4*tau2_avg)))

    cleana2 =np.clip(A2_map, None, np.percentile(A2_map, a_percentile))

    # mean weighted lifetime
    taumean_map = ((cleana1*(cleanp1**2)) + (cleana2*(cleanp2**2))) / ((cleana1*cleanp1) + (cleana2*cleanp2))


    # Phasor fit
    fr = bintime # time bin size
    t_data = np.arange(fr,fr*(spad.shape[0]+0.5),fr) #time scale for phasor plotting
    #t_data = np.linspace(fr, fr*(spad.shape[0]), num=np.int32(fr))


    freq = 1/(fr*len(t_data)) #frequency is the inverse of the decay period. This period is arbitrary, but it's best to keep it as long as the full decay  
    'this is calculating the frequency after tail fitting so window is closer to 16.5ns than 21 for 40 MHZ'
    #freq=40*1e6
    omega = 2*np.pi*freq #angular frequency = 2pi*rep rate of laser 
    'if ths is supposed to be reprate of laser why using bins after fitting?'

    #phasor calculation
    g, s = phasor_fit(spad, omega, t_data, intens, threshold=np.percentile(intens, 60))#phasor fit
    tau_pred_ratio = s/(g*omega)*1e9 #tau prediction  
    tau_pred_ratio[np.isnan(tau_pred_ratio)] = 0 #set nans to 0
    tau_pred_ratio[np.isinf(tau_pred_ratio)] = 0 #set infitities to 0
    tau_pred_ratio[tau_pred_ratio<0] = 0 # set anything below 0 to 0
    #tau_pred_ratio[intens< threshold] = float('nan')


    plot = PhasorPlot()
    plot.hist2d(g, s, cmap='plasma')
    plot.show()


    # Visualise
    phasor_taumin =  np.percentile(tau_pred_ratio,2)
    phasor_taumax =  np.percentile(tau_pred_ratio,98)


    # Plot tau based lifetime
    plt.figure()
    taus = tau_pred_ratio.copy()
    tau_threshold=np.percentile(intens,0)

    vmin = 0
    vmax = 6
    taus[intens<tau_threshold] = float('nan')

    plt.imshow(taus,vmin=vmin,vmax=vmax,cmap='plasma')
    plt.colorbar()
    plt.title(f'Phasor based lifetimes')
    plt.show()


    # Intensity from the data rather than calculating by summing unsure of what unit of "intensity acc is"

    I_flim=f['FLIM']['Intensity']

    Int = np.zeros(((I_flim[0]),len(I_flim))) # create an array with 0s the same shape as histogram data

    for i in range(len(I_flim)):
        Int[:,i] = I_flim[i] # for each value in flim allocate to Int variable 


    width = int(len(f_flim)/192) # divide length of shape by height for width
    Int = Int.reshape((-1,192,126)) # Reshape Int

    Int_og = Int.copy()
    Int = np.repeat(Int, 2, axis=-1)

    # averageing out image stack and rotating
    mean_int = np.mean(Int, axis=0)
    mean_flip = np.rot90(mean_int, k=1)


    # plotting things of interest

    # setting roi to find mean
    minrow = 100
    maxrow = minrow+20
    mincol = 120
    maxcol = mincol+20

    # all the things i want to plot and cycle through
    # definitely a better way to do this with dictionaries tbh

    names_offigs = ['Raw Intensity', 'Calculated Intensity', 'Tau 1', 'Tau 2', 
                    'Fraction a1', 'Fraction a2', 'Mean tau', 'Reduced Chi Squared']
    units_offigs = ['counts', 'counts', 'ns', 'ns', '', '', 'ns', '']
    cmaps = ['viridis', 'viridis', 'turbo', 'turbo', 'plasma', 'plasma', 'turbo', 'viridis']
    figs_ofinterest = [mean_int, intens, cleanp1, cleanp2, 
                    cleana1, cleana2, taumean_map, chi_map]
    max_vals = [None, None, 1, 5, 1, 1, 4, 4]
    min_vals = [None, None, None, None, 0, 0, None, None]
    cut = [0, 0, 0.045, 0.045, 0, 0, 0.045, 0]



    plt.figure(figsize=(22, 10))
    plt.subplots_adjust(hspace=0.2)

    for i in range(len(figs_ofinterest)):
        vals = []
        fig = figs_ofinterest[i]
        roi =fig[minrow:maxrow, mincol:maxcol]
        val = np.nanmean(roi[roi>cut[i]])
        valunc = np.std(roi[roi>cut[i]])
        vals.append(val)
        print(f'{names_offigs[i]} mean in ROI: {val:.4g} pm {valunc:.4g} {units_offigs[i]}')

        ax = plt.subplot(2, 4, i+1)
        im = ax.imshow(fig, cmap=cmaps[i], vmax=max_vals[i], vmin=min_vals[i])
        cb = plt.colorbar(im,ax=ax)
        cb.set_label(f'{units_offigs[i]}')
        plt.title(f'{names_offigs[i]}')
        rect = Rectangle((mincol, minrow), width=maxcol-mincol, height=maxrow-minrow,
                    ec='white', fill=False, linewidth=1)
        ax.add_patch(rect)
        text = f'mean in ROI:\n{val:.4g} +- {valunc:.4g}'
        ax.text(x=8, y=179, s=text, bbox=dict(facecolor='white', alpha=0.5, linestyle=None))
        ax.set_axis_off()

    plt.savefig(loc+named+'maps')

    for i in range(len(figs_ofinterest)):
        fig = figs_ofinterest[i]
        max = np.nanmax(fig)
        scale = 1/max
        img_scaled = fig * scale
        rgb = color_mapping(img_scaled)[:, :, :3]
        int_norm = intens / np.max(intens)
        for z in range(3): rgb[:, :, z] *= int_norm
        fig2, ax2= plt.subplots(figsize=(7,5))
        im = ax2.imshow(rgb, cmap='turbo', vmax=max_vals[i])
        cb = plt.colorbar(im, ax=ax2, location='bottom', fraction=0.062, pad=0.00,
                    label=f'{names_offigs[i]} ({units_offigs[i]})')
        ax2.set_axis_off()
        plt.savefig(loc+named+f'{names_offigs[i]}')


    # violin plot of lifetimes
    cleanp1[cleanp1<45e-3]=np.nan
    cleanp2[cleanp2<45e-3]=np.nan

    tau1_range = []
    tau2_range = []


    tau1_values = cleanp1[~np.isnan(cleanp1)]
    tau1_range.append(tau1_values)
    tau2_values = cleanp2[~np.isnan(cleanp2)]
    tau2_range.append(tau2_values)

    if np.size(tau1_range) > 0 and np.size(tau2_range) > 0:
        pos1 = [1]
        pos2 = [2]
        fig, ax1 = plt.subplots(figsize=(6,4))
        # plt.suptitle(f'Bound to {upper_bounds[2:4]}', y=1.01, fontweight='bold')
        vp1 = ax1.violinplot(tau1_range, pos1, showmeans=True, showextrema=True,
                    showmedians=True, widths=1,)
        for body in vp1['bodies']:
            body.set_facecolor('blue')

        ax1.get_xaxis().set_visible(False)
        ax1.set_ylabel('Tau 1 (ns)', fontsize=15)
        ax1.set_title('Calculated tau 1 and 2 values', fontsize=15)
        ax1.tick_params(axis='y', labelcolor='blue')
        # ax1.hlines(0.1, xmin=0, xmax=1)

        ax2 = ax1.twinx()
        vp2 = ax2.violinplot(tau2_range, pos2, showmeans=True, showextrema=True,
                    showmedians=True, widths=1)
        for body in vp2['bodies']:
            body.set_facecolor('darkorange')
        ax2.get_xaxis().set_visible(False)
        ax2.set_ylabel('Tau 2 (ns)', fontsize=15)
        ax2.tick_params(axis='y', labelcolor='darkorange')
        # ax2.hlines(1, xmin=1, xmax=2)
        # ax2.hlines(4, xmin=1, xmax=2)

        plt.legend([vp1['bodies'][0],vp2['bodies'][0]], ['tau 1', 'tau 2'])
        plt.savefig(loc+named+'dists')
    else:
        print('not enough data for distribution of fits, sorry boss')

    # making nice looking lifetime figures

    fig, ax = plt.subplots(figsize=(7,5))
    im = ax.imshow(taumean_map, cmap='turbo', vmax=10)
    cb = plt.colorbar(im, ax=ax, location='bottom', fraction=0.0615, pad=0,
                    label='ns')
    ax.set_axis_off()
    ax.set_title('Mean Lifetime')
    #
    np.save(loc+named+'taumean', taumean_map)
    plt.savefig(loc+named+'meanmap')

    # another nice figure

    fig = plt.figure(figsize=(9, 8), layout='constrained')
    outer = fig.add_gridspec(
        2, 2,
        width_ratios=[1, 1],
        height_ratios=[1, 1],
        wspace=0,
        hspace=0.2,
    )

    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])

    inner = outer[1, 0].subgridspec(2, 1, height_ratios=[4,1])
    ax_c = fig.add_subplot(inner[0, 0])
    ax_e = fig.add_subplot(inner[1, 0])
    ax_d = fig.add_subplot(outer[1, 1])

    ax_a.imshow(intens, cmap='gray')
    ax_a.set_axis_off()
    ax_a.annotate('(a)', (5,20), color='white', fontsize=20)

    bf_crop = np.rot90(plt.imread(r'C:\Users\laure\OneDrive - University of Glasgow\Masters\Project\Data\3107'+"/"+named+'bf.png'), k=2)[900:1350, 900:1500]
    ax_b.imshow(bf_crop)
    ax_b.set_axis_off()
    ax_b.annotate('(b)', (5,50), color='white', fontsize=20)

    ax_c.scatter(nano_tail_time, Pix2, label='pixel data', marker='x')
    ax_c.plot(nano_tail_time, fit2, color='darkorange', label=f'fit', linewidth=2)
    ax_c.legend(fontsize=12)
    ax_c.set_xlabel('Time (ns)')
    ax_c.annotate('(c)', (3, np.max(Pix2)-8), color='black', fontsize=20)

    ax_e.plot(nano_tail_time,residuals(Pix, fit), alpha=1)
    ax_e.set_ylabel('Residuals', fontsize=12)
    ax_e.get_xaxis().set_visible(False)

    im = ax_d.imshow(taumean_map, cmap='turbo', vmax=10)
    ax_d.set_axis_off()
    cb = plt.colorbar(im, ax=ax_d, location='bottom',  aspect=8.7,
                    label='(ns)')
    ax_d.annotate('(d)', (5,20), color='white', fontsize=20)

    plt.savefig(loc+named+'summary')




# %% the post processing post processing

# removing poorly fitted crap
low = 0.2
high = 8

control_summary = []
control_files = [ 'c_3_taumean.npy', 'c_1_taumean.npy', 
                 'c_4_taumean.npy', 'c_6_taumean.npy', 
                 'c_8_taumean.npy']

control_data = np.zeros((len(control_files), taumean_map.shape[0], taumean_map.shape[1]))
control_range = []
for i, file in enumerate(control_files):
    arr = np.load(loc+control_files[i])
    arr = arr[(arr > low) & (arr < high)]
    #control_data[i, :, :] = arr
    control_summary.append(np.nanmedian(arr))
    control_range.append(arr)
control_range = np.concatenate(control_range).ravel()


m10_summary = []
m10_files = ['10m_1_taumean.npy', '10m_2_taumean.npy', '1h_1_taumean.npy',
             '10m_4_taumean.npy', '10m_5_taumean.npy']
m10_range = []
m10_data = np.zeros((len(m10_files), taumean_map.shape[0], taumean_map.shape[1]))
for i, file in enumerate(m10_files):
    arr = np.load(loc+m10_files[i])
    arr = arr[(arr > low) & (arr < high)]
    #m10_data[i, :, :] = arr
    m10_summary.append(np.nanmedian(arr))
    m10_range.append(arr)
m10_range = np.concatenate(m10_range).ravel()


h24_summary = []
h24_files = ['24h_2_taumean.npy',
             '24h_4_taumean.npy', '24h_5_taumean.npy', '24h_6_taumean.npy', 
             '24h_7_taumean.npy']
h24_data = np.zeros((len(h24_files), taumean_map.shape[0], taumean_map.shape[1]))
h24_range = []
for i, file in enumerate(h24_files):
    arr = np.load(loc+h24_files[i])
    arr = arr[(arr > low) & (arr < high)]
    #h24_data[i, :, :] = arr
    h24_summary.append(np.nanmedian(arr))
    h24_range.append(arr)
h24_range = np.concatenate(h24_range).ravel()


pos1 = [1]
pos2 = [2]
pos3 = [3]
w = 1


def ptest(data1, data2, ax, range, num):
    tstat, pval = stats.mannwhitneyu(data1, data2, method='auto', alternative='greater', nan_policy='omit')

    if pval < 0.05:
        ax.annotate('*', (1.5+num, np.max(range)+0.5+num), fontsize=20)
    # elif pval < 0.1:
    #     ax.annotate('**', (1.5+num, np.max(range)+0.5+num), fontsize=20)
    # elif pval < 0.5:
    #     ax.annotate('*', (1.5+num, np.max(range)+0.5+num), fontsize=20)
    else:
        ax.annotate('-', (1.5+num, np.max(range)+0.6+num), fontsize=20)
    print('data1 p=', pval)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), width_ratios=[2,1])
fig.suptitle('Impact of rotenone treatment on mean lifetime', fontsize=25, fontweight='bold')


vp1 = ax1.violinplot(control_range, pos1, showmeans=False, showextrema=False,
               showmedians=False, widths=w)
ax1.set_ylabel('Mean lifetime (ns)', fontsize=20)
#s1 = ax1.scatter(np.ones(len(control_data)), control_summary, marker='x', s=200)
m1 = ax1.errorbar(1, np.nanmean(control_summary), yerr=np.std(control_summary), color='b', marker='o', markersize=10, capsize=5)

vp2 = ax1.violinplot(m10_range, pos2, showmeans=False, showextrema=False,
               showmedians=False, widths=w)
ptest(control_summary, m10_summary, ax1, control_range, 0)
#s2 = ax1.scatter(np.ones(len(m10_data))*2, m10_summary, marker='x', s=200)
m2 = ax1.errorbar(2, np.nanmean(m10_summary), yerr=np.std(m10_summary), color='darkorange', marker='o', markersize=10, capsize=5)

vp3 = ax1.violinplot(h24_range, pos3, showmeans=False, showextrema=False,
               showmedians=False, widths=w)
ptest(control_summary, h24_summary, ax1, control_range, 0.5)
#s3 = ax1.scatter(np.ones(len(h24_data))*3, h24_summary, marker='x', s=200)
m3 = ax1.errorbar(3, np.nanmean(h24_summary), yerr=np.nanstd(h24_summary), color='grey', marker='o', markersize=10, capsize=5)






ax1.set_ylim(None, np.max(control_range)+2)
ax1.set_xticks([1,2,3])
ax1.set_xticklabels(['control', '10 mins', '24 hrs'], size=20)

leg1 = ax2.legend([vp1['bodies'][0], vp2['bodies'][0], vp3['bodies'][0]], ['control', '10m', '24hr'], title_fontsize=20,
                  title='all pixels in all images', fontsize=20,
                  bbox_to_anchor=(0.85, 0.85), bbox_transform=fig.transFigure)

leg2 = ax2.legend([m1, m2, m3], [f'control, n={len(control_files)}',f'10m, n={len(m10_files)}', f'24hr, n={len(h24_files)}'], 
                  title_fontsize=20, fontsize=20, title='image medians',
                   bbox_to_anchor=(0.81, 0.6), bbox_transform=fig.transFigure)

ax2.add_artist(leg1)
ax2.add_artist(leg2)

ax1.hlines(np.max(control_range)+0.5, xmin=1, xmax=2, color='black')
ax1.hlines(np.max(control_range)+1, xmin=1, xmax=3, color='black')

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)


plt.show()
print(np.mean(control_summary), np.mean(m10_summary), np.nanmean(h24_summary))
print(np.nanstd(control_summary), np.nanstd(m10_summary), np.nanstd(h24_summary))

# %%


for file in control_files:
    arr = np.load(loc + file)
    arr = arr[(arr > 0) & (arr < 5)]
    control_summary.append(np.median(arr))

m10_summary = []
for file in m10_files:
    arr = np.load(loc + file)
    arr = arr[(arr > 0) & (arr < 5)]
    m10_summary.append(np.median(arr))


# %%
