import os
import glob
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt

# List analysis params
trial_length = 500
stimulus_start = 120

# List relevant data locations: these are for KAMPFF-LAB-VIDEO
root_folder = r"\\Diskstation\SurprisingMinds"
day_folder = os.path.join(root_folder, "SurprisingMinds_2017-10-21")
analysis_folder = os.path.join(day_folder, "Analysis")
csv_folder = os.path.join(analysis_folder, "csv")

# List all csv trial files
trial_files = glob.glob(csv_folder + r"/*.csv")
num_trials = len(trial_files)

# Build data matrix
data = np.empty((num_trials, trial_length))
data[:] = np.nan

# Load all trial files
index = 0
for trial_file in trial_files:
    trial = np.genfromtxt(trial_file, dtype=np.float, delimiter=",")

    # Find count of bad measurements
    bad_count = np.sum(trial < 0)

    # Measure pupil size as percentage
    trial[trial < 1] = np.nan
    # extract pupil sizes
    pupil_position = np.empty(trial_length)
    for i in range(len(trial)):
        pupil_position[i] = trial[i][0]
    # take the mean of pupil size during the whole trial
    #pupil_position_mean = np.nanmean(pupil_position)
    # make pupil size at each frame a percentage of the mean
    #pupil_position = pupil_position - pupil_position_mean

    # Only include good measurement trials with not too small pupils
    if (bad_count < (trial_length / 2)):
        data[index, :] = pupil_position
    index = index + 1

# Compute mean
data_mean = np.nanmean(data, 0)

# Plot
plt.figure()
plt.plot(data.T, color=[0.0, 0.0, 0.0, 0.1])
plt.plot(data_mean, linewidth=4, color=[1.0, 0.0, 0.0, 1.0])
plt.show()

#FIN