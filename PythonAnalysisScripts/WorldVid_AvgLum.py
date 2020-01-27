import pdb
import os
import glob
import cv2
import datetime
import math
import sys
import itertools
import csv
import fnmatch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import defaultdict
from scipy import signal
from itertools import groupby
from operator import itemgetter

### FUNCTIONS ###
def load_avg_world_unraveled(avg_world_folder_path):
    # List all world camera csv files
    stim_files = glob.glob(avg_world_folder_path + os.sep + "*Avg-World-Vid-tbuckets.csv")
    world_vids_tbucketed = {}
    for stim_file in stim_files:
        stim_filename = stim_file.split(os.sep)[-1]
        stim_type = stim_filename.split('_')[1]
        stim_number = stim_name_to_float[stim_type]
        world_vids_tbucketed[stim_number] = {}
        extracted_rows = []
        print("Extracting from {name}".format(name=stim_filename))
        with open(stim_file) as f:
            csvReader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            for row in csvReader:
                extracted_rows.append(row)
        print("Unraveling average frame data...")
        for i in range(len(extracted_rows)):
            if i==0:
                unravel_height = int(extracted_rows[i][0])
                unravel_width = int(extracted_rows[i][1])
                world_vids_tbucketed[stim_number]["Vid Dimensions"] = [unravel_height, unravel_width]
            elif i==1:
                vid_count = int(extracted_rows[i][0])
                world_vids_tbucketed[stim_number]["Vid Count"] = vid_count
            else:
                tbucket_num = extracted_rows[i][0]
                flattened_frame = extracted_rows[i][1:]
                flat_frame_array = np.array(flattened_frame)
                unraveled_frame = np.reshape(flat_frame_array,(unravel_height,unravel_width))
                world_vids_tbucketed[stim_number][tbucket_num] = unraveled_frame
    return world_vids_tbucketed

def downsample_avg_world_vids(unraveled_world_vids_dict, original_bucket_size_ms, new_bucket_size_ms):
    if (new_bucket_size_ms % original_bucket_size_ms == 0):
        new_sample_rate = int(new_bucket_size_ms/original_bucket_size_ms)
        downsampled_world_vids_dict = {}
        for stim in unraveled_world_vids_dict.keys():
            print("Working on stimulus {s}".format(s=stim))
            downsampled_world_vids_dict[stim] = {}
            vid_metadata_keys = sorted([x for x in unraveled_world_vids_dict[stim].keys() if type(x) is str])
            for metadata in vid_metadata_keys:
                downsampled_world_vids_dict[stim][metadata] = unraveled_world_vids_dict[stim][metadata]
            this_stim_avg_vid_dimensions = unraveled_world_vids_dict[stim][vid_metadata_keys[1]]
            tbuckets = sorted([x for x in unraveled_world_vids_dict[stim].keys() if type(x) is float])
            padding = new_sample_rate - (int(tbuckets[-1]) % new_sample_rate)
            original_tbuckets_sliced = range(0, int(tbuckets[-1]+padding), new_sample_rate)
            new_tbucket = 0
            for i in original_tbuckets_sliced:
                start = i
                end = i + new_sample_rate - 1
                this_slice_summed_frame = np.zeros((this_stim_avg_vid_dimensions[0], this_stim_avg_vid_dimensions[1]))
                this_slice_tbuckets = []
                this_slice_count = 0
                for tbucket in tbuckets:
                    if start<=tbucket<=end:
                        this_slice_tbuckets.append(tbucket)
                for bucket in this_slice_tbuckets:
                    this_slice_summed_frame = this_slice_summed_frame + unraveled_world_vids_dict[stim][bucket]
                    this_slice_count = this_slice_count + 1
                this_slice_avg_frame = this_slice_summed_frame/float(this_slice_count)
                downsampled_world_vids_dict[stim][new_tbucket] = this_slice_avg_frame
                new_tbucket = new_tbucket + 1
        return downsampled_world_vids_dict
    else:
        print("Sample rate must be a multiple of {bucket}".format(bucket=original_bucket_size))

### BEGIN ANALYSIS ###
# List relevant data locations: these are for KAMPFF-LAB-VIDEO
root_folder = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\dataPythonWorkflows"
# set up folders
plots_folder = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\plots"
pupils_folder = os.path.join(plots_folder, "pupil")
pupils_pv_folder = os.path.join(pupils_folder, "withPeaksValleys")
engagement_folder = os.path.join(plots_folder, "engagement")
linReg_folder = os.path.join(plots_folder, "linReg")
pooled_stim_vids_folder = os.path.join(plots_folder, "PooledStimVids")
# Create plots folder (and sub-folders) if it (they) does (do) not exist
if not os.path.exists(plots_folder):
    #print("Creating plots folder.")
    os.makedirs(plots_folder)
if not os.path.exists(pupils_folder):
    #print("Creating camera profiles folder.")
    os.makedirs(pupils_folder)
if not os.path.exists(pupils_pv_folder):
    #print("Creating camera profiles folder.")
    os.makedirs(pupils_pv_folder)
if not os.path.exists(engagement_folder):
    #print("Creating engagement count folder.")
    os.makedirs(engagement_folder)
if not os.path.exists(linReg_folder):
    #print("Creating engagement count folder.")
    os.makedirs(linReg_folder)
if not os.path.exists(pooled_stim_vids_folder):
    #print("Creating engagement count folder.")
    os.makedirs(pooled_stim_vids_folder)

### TIMING/SAMPLING VARIABLES FOR DATA EXTRACTION
# downsample = collect data from every 40ms or other multiples of 20
downsampled_bucket_size_ms = 40
original_bucket_size_in_ms = 4
max_length_of_stim_vid = 60000 # milliseconds
no_of_time_buckets = max_length_of_stim_vid/original_bucket_size_in_ms
downsampled_no_of_time_buckets = max_length_of_stim_vid/downsampled_bucket_size_ms
new_time_bucket_sample_rate = downsampled_bucket_size_ms/original_bucket_size_in_ms
milliseconds_for_baseline = 3000
baseline_no_buckets = int(milliseconds_for_baseline/new_time_bucket_sample_rate)
### STIMULI VID INFO
stim_vids = [24.0, 25.0, 26.0, 27.0, 28.0, 29.0]
stim_name_to_float = {"Stimuli24": 24.0, "Stimuli25": 25.0, "Stimuli26": 26.0, "Stimuli27": 27.0, "Stimuli28": 28.0, "Stimuli29": 29.0}
stim_float_to_name = {24.0: "Stimuli24", 25.0: "Stimuli25", 26.0: "Stimuli26", 27.0: "Stimuli27", 28.0: "Stimuli28", 29.0: "Stimuli29"}


### BEGIN MONTHLY AVERAGE DATA EXTRACTION ###
all_months_avg_world_vids = {}
### EXTRACT, UNRAVEL, SAVE TO FILE TIME BINNED STIM VIDEOS ###
# update list of completed world vid average folders on dropbox
day_folders = sorted(os.listdir(root_folder))
avg_world_vid_folders = fnmatch.filter(day_folders, 'WorldVidAverage_*')
updated_folders_to_extract = []
for avg_world_vid_folder in avg_world_vid_folders:
    folder_year_month = avg_world_vid_folder.split('_')[1]
    if folder_year_month not in all_months_avg_world_vids.keys():
        updated_folders_to_extract.append(avg_world_vid_folder)

# extract, unravel, write to video, and save
for month_folder in updated_folders_to_extract:
    month_name = month_folder.split('_')[1]
    all_months_avg_world_vids[month_name] = {}
    month_folder_path = os.path.join(root_folder, month_folder)
    # unravel
    unraveled_monthly_world_vids = load_avg_world_unraveled(month_folder_path)
    # downsample
    print("Downsampling monthly averaged stimulus videos for {month}".format(month=month_name))
    downsampled_monthly_world_vids = downsample_avg_world_vids(unraveled_monthly_world_vids, original_bucket_size_in_ms, downsampled_bucket_size_ms)
    # now need to convert these frame arrays into luminance value, one per timebucket

    
### END MONTHLY AVERAGE DATA EXTRACTION ###

stim24_frames = downsampled_monthly_world_vids[24.0]
lums_stim24 = []
for key in stim24_frames:
    if key == 'Vid Count' or key == 'Vid Dimensions':
        continue
    else:
        frame = stim24_frames[key]
        lum = np.mean(frame[:])
        lums_stim24.append(lum)
lums_stim24 = np.array(lums_stim24)
    
