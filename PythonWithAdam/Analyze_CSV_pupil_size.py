import os
import glob
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt

# List analysis params
clip_length = 1000 # frames
stimulus_start = 120

# List relevant data locations: these are for KAMPFF-LAB-VIDEO
root_folder = r"C:\Users\KAMPFF-LAB-VIDEO\Dropbox\SurprisingMinds\analysis\pythonWithAdam-csv"

# consolidate csv files from multiple days into one data structure

# for each day...
day_folder = os.path.join(root_folder, "SurprisingMinds_2017-10-21")
analysis_folder = os.path.join(day_folder, "Analysis")
csv_folder = os.path.join(analysis_folder, "csv")

# List all csv trial files
eye = "right"
right_trial_files = glob.glob(csv_folder + os.sep + eye + "*.csv")
left_trial_files = glob.glob(csv_folder + r"/left*.csv")
num_right_trials = len(right_trial_files)
num_left_trials = len(left_trial_files)

# Print/save number of users per day
folder_name = day_folder.split(os.sep)[-1]
day_name = folder_name.split("_")[-1]
print("On {day}, exhibit was activated {right_count} times (based on right eye camera activation; left camera activated {left_count} times)".format(day=day_name, right_count=num_right_trials, left_count=num_left_trials))


# for each day...
def load_daily_eye_trials(which_eye, day_csv_files, trial_length): 
    # List all csv trial files
    trial_files = glob.glob(day_csv_files + os.sep + which_eye + "*.csv")
    num_trials = len(trial_files)
    # build array for resultant data
    data = np.empty((num_trials, trial_length))
    data[:] = np.nan
    

# Build data matrix
right_data = np.empty((num_right_trials, clip_length))
right_data[:] = np.nan

left_data = np.empty((num_left_trials, clip_length))
left_data[:] = np.nan
# columns = time points
# rows = each person (trial)
## when doing this, filter for bad trials
# decide criteria for pre-processing the data
## COMBINE EXTRACTING PUPIL SIZE AND POSITION

# Load all trial files
index = 0
for trial_file in trial_files:
    trial = np.genfromtxt(trial_file, dtype=np.float, delimiter=",")

    # Find count of bad measurements
    bad_count = np.sum(trial < 0)

    # Measure pupil size as percentage
    trial[trial < 0] = np.nan
    # extract pupil sizes
    pupil_sizes = np.empty(clip_length)
    for i in range(len(trial)):
        pupil_sizes[i] = trial[i][2]
    # take the mean of pupil size during the whole trial
    pupil_sizes_mean = np.nanmean(pupil_sizes)
    # make pupil size at each frame a percentage of the mean
    pupil_sizes = (pupil_sizes / pupil_sizes_mean) * 100

    # Only include good measurement trials with not too small pupils
    if (bad_count < (clip_length / 2)):
        data[index, :] = pupil_sizes
    index = index + 1

# Compute mean
data_mean = np.nanmean(data, 0)

# Plot
plt.figure()
plt.plot(data.T, '.', color=[0.0, 0.0, 0.0, 0.1])
plt.plot(data_mean, linewidth=4, color=[1.0, 0.0, 0.0, 1.0])
plt.show()

#FIN