# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:11:23 2026

@author: 2565140D

Convert data from the H5 streams into usable images


"""
#%% imports
import h5py as hp
import time
import pickle
from loguru import logger
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, ndimage
from bisect import bisect_left
import cv2
from scipy.optimize import curve_fit
#%% Constant loading
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
LOW_PIXEL_THRESHOLD = 0.25

# The default start and end crop ranges for the IRFs
DEFAULT_IRF_CROP_START = 20
DEFAULT_IRF_CROP_END = 50

#%% functions
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

def phasor_fit(data,omega,t_data,intens):
    """returns the g and s values of a phasor fit"""
    g = np.zeros((data.shape[1],data.shape[2]))
    s = np.zeros((data.shape[1],data.shape[2]))
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            if intens[i,j]!=0:
                g[i,j] = np.sum(data[:,i,j]*np.cos(omega*t_data))/intens[i,j]
                s[i,j] = np.sum(data[:,i,j]*np.sin(omega*t_data))/intens[i,j] # reason for difference between data and intensity is to be able to apply thresholding
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
    tau_map = np.zeros((data.shape[0], data.shape[1]))
    A_map = np.zeros((data.shape[0], data.shape[1]))
    offset_map = np.zeros((data.shape[0], data.shape[1]))
    ''' make zero array for each map '''
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
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
                        maxfev=10000
                    )
                    
                    A_map[i, j] = popt[0]
                    tau_map[i, j] = popt[1]
                    offset_map[i, j] = popt[2]
                    
                except RuntimeError:
                    # If fitting fails, set to NaN
                    A_map[i, j] = np.nan
                    tau_map[i, j] = np.nan
                    offset_map[i, j] = np.nan
            else:
                A_map[i, j] = np.nan
                tau_map[i, j] = np.nan
                offset_map[i, j] = np.nan
    
    return tau_map, A_map, offset_map

#%% load up data and see names
path=r"C:\Users\laure\MSc\Project\FLIM-Utility-main\Notebooks\Data\26_06\525BP\MCF_1.h5"

f=hp.File("./Data/2606/460BP/MCF_2.h5") # Load in file from path
keylist=list(f.keys())# list the keys in the file? unsure how 
dset=f[keylist[0]]

keys=list(dset.keys())
print(keys)


data=(f)["Streams"]["Photon"][()]

Metadata=dict(f["Streams"]["Photon"].attrs)  # Photon dataset attributes
bintime=Metadata["TimePerBin"][0]
nano_bintime=bintime*1e9

        
#%%from stream to dream

stream=data.flatten().astype(np.uint32)#convert to 32 bit
photon_stream=stream[stream !=0]#remove zeros
micro_times = (photon_stream >> 16) & 0xFFF# time of each photon on the pixel
y_positions = photon_stream & 0xFF
x_positions = (photon_stream >> 8) & 0xFF
markers = (photon_stream >> 28) & 0xF

#%% framing the question

frame_marker_mask = ((photon_stream >> 28) & 0xF) == 0xF
frame_marker_locations = np.nonzero(frame_marker_mask)[0]


frame_marker_info = photon_stream[frame_marker_locations]
frame_marker_types = (frame_marker_info >> 16) & 0xFFF

# We create boolean masks for the marker types
starts_mask = frame_marker_types == 0x001
ends_mask = frame_marker_types == 0x002
triggers_mask = frame_marker_types == 0x010

frame_starts = frame_marker_locations[starts_mask]
frame_ends = frame_marker_locations[ends_mask]
triggers_rising_edge = frame_marker_locations[triggers_mask]
num_frames=len(frame_starts)

#%% Things are getting intense
#set range of interest per image i.e how long sampling period  
frame_range=(0,num_frames-1)
start_index= frame_starts[frame_range[0]]
end_index=frame_starts[frame_range[1]]
x_in_range = x_positions[start_index:end_index]
y_in_range = y_positions[start_index:end_index]
markers_in_range = markers[start_index:end_index]

# Pull out only our photon events
x_in_range = x_in_range[markers_in_range == 2]
y_in_range = y_in_range[markers_in_range == 2]

# Form our intensity image
intensity_image = np.bincount(y_in_range * DIMENSION_X + x_in_range,
                              minlength=np.prod((DIMENSION_Y, DIMENSION_X))).reshape((DIMENSION_Y, DIMENSION_X))# convert to a pixel map then count intensities
plt.imshow(intensity_image)



#%% once in a Lifetime
frame_range=(0,num_frames-1)
start_index= frame_starts[frame_range[0]]
end_index=frame_starts[frame_range[1]]
x_in_range = x_positions[start_index:end_index]
y_in_range = y_positions[start_index:end_index]
markers_in_range = markers[start_index:end_index]
micro_times_in_range = micro_times[start_index:end_index]

# Pull out only our photon events
x_in_range = x_in_range[markers_in_range == 2]
y_in_range = y_in_range[markers_in_range == 2]
micro_times_in_range=micro_times_in_range[markers_in_range==2]

t_min = micro_times_in_range.min()
micro_times_in_range -= t_min
t_max = micro_times_in_range.max()
micro_time_range=1+t_max

lin = y_in_range * (DIMENSION_X * micro_time_range) + x_in_range * micro_time_range + micro_times_in_range

tcspc_image = np.bincount(lin, minlength=DIMENSION_Y *
                                                 DIMENSION_X *
                                                 micro_time_range).reshape(DIMENSION_Y, DIMENSION_X, micro_time_range)


intensity = tcspc_image.sum(axis=2)  # sum across time bins → 2D intensity image
bright_y, bright_x = np.unravel_index(intensity.argmax(), intensity.shape)

# Choose your second pixel manually
second_y, second_x = 54, 99  # change these to whatever position you want

# Time axis

time_axis = np.arange(0,(nano_bintime*(micro_time_range-1)),nano_bintime)

# Plot
plt.figure()
plt.plot(time_axis, tcspc_image[bright_y, bright_x, :], label=f'Brightest pixel ({bright_x}, {bright_y})')
plt.plot(time_axis, tcspc_image[second_y, second_x, :], label=f'Pixel 2 ({second_x}, {second_y})')
plt.xlabel('Micro time [ns]')
plt.ylabel('Photon count')
plt.legend()
plt.title('Decay curves')
plt.show()

#%% Calibration
timing_calibration=np.load(r"C:\Users\laure\MSc\Project\FLIM-Utility-main\Notebooks\Example Data\IRFCalibrationData.H5", allow_pickle=True)
Calibrated_TCSPC=adjust_timings(tcspc_image, timing_calibration)

plt.figure()
plt.plot(time_axis, Calibrated_TCSPC[bright_y, bright_x, :], label=f'Brightest pixel ({bright_x}, {bright_y})')
plt.plot(time_axis, Calibrated_TCSPC[second_y, second_x, :], label=f'Pixel 2 ({second_x}, {second_y})')
plt.xlabel('Micro time [ns]')
plt.ylabel('Photon count')
plt.legend()
plt.title('Decay curves')
plt.show()
#%% Tail extraction
decay_curve=Calibrated_TCSPC[bright_y,bright_x,:]
peak_idx = np.argmax(decay_curve)
    
# Cut from slightly before the peak
start_idx = max(0, peak_idx+2)  # max(0) stops us going negative

decay = peak_idx = decay_curve[start_idx:]
cal_decay_time = np.arange(len(decay))

plt.figure()
plt.plot(cal_decay_time, decay, label=f'Brightest pixel ({bright_x}, {bright_y})')

Cut_CalTCSPC=Calibrated_TCSPC[:,:,start_idx:]
Cut_time=time_axis[start_idx:]

plt.figure()
plt.plot(Cut_time, Cut_CalTCSPC[bright_y, bright_x, :], label=f'Brightest pixel ({bright_x}, {bright_y})')
plt.plot(Cut_time, Cut_CalTCSPC[second_y, second_x, :], label=f'Pixel 2 ({second_x}, {second_y})')
plt.xlabel('Micro time (bins)')
plt.ylabel('Photon count')
plt.legend()
plt.title('Decay curves')
plt.show()
#%%individual pixels
Pix=Cut_CalTCSPC[bright_y,bright_x]
Pix2=Cut_CalTCSPC[second_y, second_x]

A_guess = np.max(Pix)
tau_guess = Cut_time[np.argmax(Pix)] / 2
offset_guess = np.min(Pix)

# Fit the exponential decayy
popt1,pcov1 = curve_fit(
    exponential_decay, 
    Cut_time, 
    Pix,
    p0=[A_guess, tau_guess, offset_guess],
)



popt2,pcov2 = curve_fit(
    exponential_decay, 
    Cut_time, 
    Pix2,
    p0=[A_guess, tau_guess, offset_guess],
)

fit=exponential_decay(Cut_time, *popt1 )
fit2=exponential_decay(Cut_time, *popt2 )
plt.figure()
plt.scatter(Cut_time,Pix,label=('Pixel 1'))
plt.plot(Cut_time,fit,color='r',label=f'Fit 1: A=%5.3f, tau=%5.3f, Offset=%5.3f' % tuple(popt1))
plt.scatter(Cut_time,Pix2,label=('Pixel 2'))
plt.plot(Cut_time,fit2,color='b',label=f'Fit 2: A=%5.3f, tau=%5.3f, Offset=%5.3f' % tuple(popt2))
plt.legend()
plt.title('Tails')
#%%calculate full image
t_data = np.arange(bintime,bintime*(Cut_CalTCSPC.shape[0]+1),bintime)
tau_map, A_map, offset_map=fit_exponential_decay(Cut_CalTCSPC, t_data, intensity,threshold=np.percentile(intensity_image,95))
tau_map[np.isnan(tau_map)] = 0 #set nans to 0
tau_map[np.isinf(tau_map)] = 0 #set infitities to 0
tau_map[tau_map<0] = 0 # set anything below 0 to 0
#%%clean up
tau_avg = np.mean(tau_map)
tau_max= np.max(tau_map)
cleanp = np.clip(tau_map, None, np.percentile(tau_map, 99))


#%% image of both lifetime and intensity
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

im0=axes[0].imshow(cleanp,cmap='Blues')
axes[0].set_title('lifetime')
fig.colorbar(im0,ax=axes[0])
axes[0].axis('off')

im1=axes[1].imshow(intensity_image)
axes[1].set_title('intensity')
axes[1].axis('off')

plt.tight_layout()
plt.show()


