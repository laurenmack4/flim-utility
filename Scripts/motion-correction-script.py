
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from cycler import cycler
import os

from FlimeraData import FlimeraData, FlimeraCalibration, TCSPCImage, NoiseModel, FitParameters, Algorithm, LifetimeCalculator
from FlimeraData.Util.TCSPCImage import ROI, TCSPCImage
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter
import FlimeraData.MotionCorrectionFLIM as mcf

DIM_X = 192
DIM_Y = 127
WIDTH = 18.4 * 1e-6
HEIGHT = 9.2 * 1e-6

# this currently does not work

directory = r'C:\Users\2653957M\Onedrive - University of Glasgow\Data\2026\26_10\20261014/'
data_location = os.fsencode(directory)

path_to_fiji_install = r"C:\Users\2653957M\Apps\Fiji" # You must change this path to reflect where your local install is!
lifetime_calculator = LifetimeCalculator(imagej_path=path_to_fiji_install, log=False)

for file in os.listdir(data_location):
    filename = os.fsdecode(file)
    if filename.endswith('.H5'):
        flimera_data = FlimeraData(file_path=directory+filename, magnification=40, log=False)

        intensity_im = flimera_data.generate_intensity_image(remove_outliers=True)
        tcspc_im = flimera_data.generate_TCSPC_image(frame_range=[0,-1], remove_outliers_t=True)
        
        fit_start = np.median(np.argmax(tcspc_im.image, axis=2)).astype(int)
        fit_end = len(tcspc_im.inspect_histogram(100,100)[0]) - 1
        print(fit_start, fit_end)


        fit_parameters = FitParameters(noise_model=NoiseModel.POISSON_DATA, algorithm=Algorithm.GLOBAL, 
                               intensity_threshold_percent=80, lifetime_range=(0, 10e-9),  
                               fit_range=(fit_start, fit_end),num_lifetime_components=1)
        lifetime_image = lifetime_calculator.calculate_lifetimes(tcspc_image=tcspc_im, fit_parameters=fit_parameters) 

        fig, ax = plt.subplots(1, 2)
        im = ax[0].imshow(lifetime_image.average_lifetime_image*1e9, cmap='turbo', aspect=flimera_data.pixel_width/flimera_data.pixel_height)
        cb = plt.colorbar(im, ax=ax[0], location='right', fraction=0.065, pad=0,
                        label='mean lifetime (ns)')
        ax[0].set_axis_off()
        ax[0].set_title(f'Mean Lifetime')

        im = ax[1].imshow(lifetime_image.tcspc_image.intensity_image, cmap='gist_gray', aspect=flimera_data.pixel_width/flimera_data.pixel_height)
        cb = plt.colorbar(im, ax=ax[1], location='right', fraction=0.065, pad=0,
                        label='counts')
        ax[1].set_axis_off()
        ax[1].set_title(f'Intensity')
        plt.suptitle(f'Raw')
        plt.tight_layout()
        plt.savefig(directory+f'{filename[:-3]}_raw_lifetime')
        plt.close()

        marker_est = mcf.motion_correction_markers(flimera_data, lifetime_calculator, bins=1, 
                                                   velocity=-mcf.estimate_velocity_max(flimera_data, lifetime_calculator, (0, 80), binning=10, output=False), 
                                                   output=False,int_thresh_pc=80, save=True, saveloc=directory+f'{filename[:-3]}_lifetime_corrected')   
    
    else:
        pass