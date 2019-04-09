import os
import glob
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt

### FUNCTIONS ###
def load_daily_pupil_areas(which_eye, day_folder_path, trial_length): 
    # List all csv trial files
    trial_files = glob.glob(day_folder_path + os.sep + which_eye + "*.csv")
    num_trials = len(trial_files)

    data_contours = np.empty((num_trials, trial_length))
    data_contours[:] = np.nan

    data_circles = np.empty((num_trials, trial_length))
    data_circles[:] = np.nan

    index = 0
    for trial_file in trial_files:
        trial = np.genfromtxt(trial_file, dtype=np.float, delimiter=",")

        # Find count of bad measurements
        bad_count = np.sum(trial < 0)

        if (bad_count < (trial_length/2)): 
            trial[trial<0] = np.nan
            pupil_sizes_contours = np.empty(trial_length)
            pupil_sizes_circles = np.empty(trial_length)
            for i in range(len(trial)):
                pupil_sizes_contours[i] = trial[i][2]
                pupil_sizes_circles[i] = trial[i][5]
            data_contours[index, :] = pupil_sizes_contours
            data_circles[index, :] = pupil_sizes_circles
        index = index + 1
    return data_contours, data_circles

def list_sub_folders(path_to_root_folder):
    # List all sub folders
    sub_folders = []
    for folder in os.listdir(path_to_root_folder):
        if(os.path.isdir(os.path.join(path_to_root_folder, folder))):
            sub_folders.append(os.path.join(path_to_root_folder, folder))
    return sub_folders


# List analysis params
clip_length = 1000 # frames
stimulus_start = 120

# List relevant data locations: these are for KAMPFF-LAB-VIDEO
root_folder = r"C:\Users\KAMPFF-LAB-VIDEO\Dropbox\SurprisingMinds\analysis\pythonWithAdam-csv"

# consolidate csv files from multiple days into one data structure
day_folders = list_sub_folders(root_folder)

all_right_trials = []
all_left_trials = []

for day_folder in day_folders: 
    # for each day...
    day_folder_path = os.path.join(root_folder, day_folder)
    analysis_folder = os.path.join(day_folder_path, "Analysis")
    csv_folder = os.path.join(analysis_folder, "csv")

    # Print/save number of users per day
    day_name = day_folder.split("_")[-1]

    right_area_contours, right_area_circles = load_daily_pupil_areas("right", csv_folder, clip_length)
    left_area_contours, left_area_circles = load_daily_pupil_areas("left", csv_folder, clip_length)

    print("On {day}, exhibit was activated {count} times".format(day=day_name, count=len(right_area_contours)))

    ## COMBINE EXTRACTING PUPIL SIZE AND POSITION

    # filter data
    right_area_contours[right_area_contours>15000] = np.nan
    left_area_contours[left_area_contours>15000] = np.nan

    # create a baseline - take first 3 seconds, aka 180 frames
    right_area_contours_baseline = np.nanmedian(right_area_contours[:,0:180], 1)
    left_area_contours_baseline = np.nanmedian(left_area_contours[:,0:180], 1)

    # normalize and append
    for index in range(len(right_area_contours_baseline)): 
        right_area_contours[index,:] = right_area_contours[index,:]/right_area_contours_baseline[index]
        all_right_trials.append(right_area_contours[index,:])
    for index in range(len(left_area_contours_baseline)): 
        left_area_contours[index,:] = left_area_contours[index,:]/left_area_contours_baseline[index]
        all_left_trials.append(left_area_contours[index,:])




all_right_trials_array = np.array(all_right_trials)
all_left_trials_array = np.array(all_left_trials)     
# Compute global mean
all_right_contours_mean = np.nanmean(all_right_trials_array, 0)
all_left_contours_mean = np.nanmean(all_left_trials_array, 0)








# Plot
plt.figure()

plt.subplot(2,1,1)
plt.plot(all_right_trials_array.T, '.', MarkerSize=1, color=[0.0, 0.0, 0.0, 0.025])
plt.plot(all_right_contours_mean, linewidth=4, color=[1.0, 0.0, 0.0, 0.3])
plt.ylim(0,2)

plt.subplot(2,1,2)
plt.plot(all_left_trials_array.T, '.', MarkerSize=1, color=[0.0, 0.0, 0.0, 0.025])
plt.plot(all_left_contours_mean, linewidth=4, color=[1.0, 0.0, 0.0, 0.3])
plt.ylim(0,2)

plt.show()

#FIN