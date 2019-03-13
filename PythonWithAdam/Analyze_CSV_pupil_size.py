import os
import glob
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt

# List analysis params
trial_length = 360
stimulus_start = 120

# List relevant data locations
root_folder = "/home/kampff/DK"
day_folder = root_folder + "/SurpriseIntelligence_2017-07-25"
analysis_folder = day_folder + "/Analysis"
csv_folder = analysis_folder + "/csv_pupil_size"

# List all csv trial files
trial_files = glob.glob(csv_folder + '/*.csv')
num_trials = len(trial_files)

# Build data matrix
data = np.empty((num_trials, trial_length))
data[:] = np.nan

# Load all trial files
index = 0
for trial_file in trial_files:
    trial = np.genfromtxt(trial_file, dtype=np.float)

    # Find count of bad measurements
    bad_count = np.sum(trial < 0)

    # Measure pupil size as percentage
    trial[trial < 0] = np.nan
    trial_mean = np.nanmean(trial)
    trial = (trial / trial_mean) * 100

    # Only include good measurement trials with not too small pupils
    if (bad_count < (trial_length / 2)):
        data[index, :] = trial
    index = index + 1

# Compute mean
data_mean = np.nanmean(data, 0)

# Plot
plt.figure()
plt.plot(data.T, '.', color=[0.0, 0.0, 0.0, 0.1])
plt.plot(data_mean, linewidth=4, color=[1.0, 0.0, 0.0, 1.0])
plt.show()

#FIN