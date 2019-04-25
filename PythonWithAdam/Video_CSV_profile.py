import os
import glob
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import shutil
import fnmatch
import sys
import math

data_drive = r"C:\Users\taunsquared\Documents\GitHub\SurprisingMinds-Analysis\PythonWithAdam\temp"

def time_between_frames(timestamps_csv):
    time_diffs = []
    for t in range(len(timestamps_csv)):
        this_timestamp = timestamps_csv[t].split('+')[0][:-1]
        this_time = datetime.datetime.strptime(this_timestamp, "%Y-%m-%dT%H:%M:%S.%f")
        if (t==0):
            time_diffs.append(0)
            last_time = this_time
        else: 
            time_diff = this_time - last_time
            time_diff_seconds = time_diff.total_seconds()
            time_diffs.append(time_diff_seconds)
            last_time = this_time
    return np.array(time_diffs)

def find_target_frame(ref_timestamps_csv, target_timestamps_csv, ref_frame):
    # Find the frame in one video that best matches the timestamp of ref frame from another video
    # Get ref frame time
    ref_timestamp = ref_timestamps_csv[ref_frame]
    ref_timestamp = ref_timestamp.split('+')[0][:-1]
    ref_time = datetime.datetime.strptime(ref_timestamp, "%Y-%m-%dT%H:%M:%S.%f")
    # Generate delta times (w.r.t. start_frame) for every frame timestamp
    frame_counter = 0
    for timestamp in target_timestamps_csv:
        timestamp = timestamp.split('+')[0][:-1]
        time = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f")
        timedelta = ref_time - time
        seconds_until_alignment = timedelta.total_seconds()
        if(seconds_until_alignment < 0):
            break
        frame_counter = frame_counter + 1
    return frame_counter

def list_sub_folders(path_to_root_folder):
    # List all sub folders
    sub_folders = []
    for folder in os.listdir(path_to_root_folder):
        if(os.path.isdir(os.path.join(path_to_root_folder, folder))):
            sub_folders.append(os.path.join(path_to_root_folder, folder))
    return sub_folders

# List all trial folders
day_folder = r"C:\Users\KAMPFF-LAB-VIDEO\Documents\GitHub\SurprisingMinds-Analysis\PythonWithAdam\temp"
trial_folders = list_sub_folders(day_folder)
num_trials = len(trial_folders)

trial_folder = trial_folders[0]
trial_name = trial_folder.split(os.sep)[-1]
# Load CSVs and create timestamps
# ------------------------------
#print("Loading csv files for {trial}...".format(trial=trial_name))
# Get world movie timestamp csv path
world_csv_path = glob.glob(trial_folder + '/*world.csv')[0]
stimuli_number = world_csv_path.split("_")[-2]

# create dictionary of start frames for octopus clip
octo_frames = {"stimuli024": 438, "stimuli025": 442, "stimuli026": 517, "stimuli027": 449, "stimuli028": 516, "stimuli029": 583}
world_octo_start = octo_frames[stimuli_number]

# Load world CSV
world_timestamps = np.genfromtxt(world_csv_path, dtype=np.str, delimiter=' ')

# Get eye timestamp csv paths
right_eye_csv_path = glob.glob(trial_folder + '/*righteye.csv')[0]
left_eye_csv_path = glob.glob(trial_folder + '/*lefteye.csv')[0]

# Load eye CSVs
right_eye_timestamps = np.genfromtxt(right_eye_csv_path, dtype=np.str, delimiter=' ')
left_eye_timestamps = np.genfromtxt(left_eye_csv_path, dtype=np.str, delimiter=' ')

# trim csvs to just octopus video
right_octo = find_target_frame(world_timestamps, right_eye_timestamps, world_octo_start)
left_octo = find_target_frame(world_timestamps, left_eye_timestamps, world_octo_start)

world_octo_timestamps = world_timestamps[world_octo_start:]
right_octo_timestamps = right_eye_timestamps[right_octo:]
left_octo_timestamps = left_eye_timestamps[left_octo:]

# Generate delta times (w.r.t. start_frame) for every frame timestamp
right_time_diffs_array = time_between_frames(right_eye_timestamps)
left_time_diffs_array = time_between_frames(left_eye_timestamps)
world_time_diffs_array = time_between_frames(world_timestamps)

plt.figure()

plt.subplot(3,1,1)
plt.plot(right_time_diffs_array.T,'.', MarkerSize=1, color=[1.0, 0.0, 0.0, 0.7])

plt.subplot(3,1,2)
plt.plot(left_time_diffs_array.T,'.', MarkerSize=1, color=[0.0, 1.0, 0.0, 0.7])

plt.subplot(3,1,3)
plt.plot(world_time_diffs_array.T, '.', MarkerSize = 1, color=[0.0, 0.0, 1.0, 0.7])

plt.show()
