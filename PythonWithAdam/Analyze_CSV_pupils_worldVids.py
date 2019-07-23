### --------------------------------------------------------------------------- ###
# this script uses ImageMagick to easily install ffmpeg onto Windows 10: 
# https://www.imagemagick.org/script/download.php
### --------------------------------------------------------------------------- ###
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
from scipy.signal import savgol_filter
from itertools import groupby
from operator import itemgetter
from scipy.signal import find_peaks
### FUNCTIONS ###
def load_daily_pupils(which_eye, day_csv_folder_path, max_no_of_buckets, original_bucket_size, new_bucket_size): 
    if (new_bucket_size % original_bucket_size == 0):
        new_sample_rate = int(new_bucket_size/original_bucket_size)
        max_no_of_buckets = int(max_no_of_buckets)
        #print("New bucket window = {size}, need to average every {sample_rate} buckets".format(size=new_bucket_size, sample_rate=new_sample_rate))
        # List all csv trial files
        trial_files = glob.glob(day_csv_folder_path + os.sep + which_eye + "*.csv")
        num_trials = len(trial_files)
        good_trials = num_trials
        # contours
        data_contours_X = np.empty((num_trials, max_no_of_buckets+1))
        data_contours_X[:] = -6
        data_contours_Y = np.empty((num_trials, max_no_of_buckets+1))
        data_contours_Y[:] = -6
        data_contours = np.empty((num_trials, max_no_of_buckets+1))
        data_contours[:] = -6
        # circles
        data_circles_X = np.empty((num_trials, max_no_of_buckets+1))
        data_circles_X[:] = -6
        data_circles_Y = np.empty((num_trials, max_no_of_buckets+1))
        data_circles_Y[:] = -6
        data_circles = np.empty((num_trials, max_no_of_buckets+1))
        data_circles[:] = -6

        index = 0
        for trial_file in trial_files:
            trial_name = trial_file.split(os.sep)[-1]
            trial_stimulus = trial_name.split("_")[1]
            trial_stim_number = np.float(trial_stimulus[-2:])
            trial = np.genfromtxt(trial_file, dtype=np.float, delimiter=",")
            # if there are too many -5 rows (frames) in a row, don't analyse this trial
            bad_frame_count = []
            for frame in trial:
                if frame[0]==-5:
                    bad_frame_count.append(1)
                else:
                    bad_frame_count.append(0)
            clusters =  [(x[0], len(list(x[1]))) for x in itertools.groupby(bad_frame_count)]
            longest_cluster = 0
            for cluster in clusters:
                if cluster[0] == 1 and cluster[1]>longest_cluster:
                    longest_cluster = cluster[1]
            #print("For trial {name}, the longest cluster is {length}".format(name=trial_name, length=longest_cluster))
            if longest_cluster<100:
                no_of_samples = math.ceil(len(trial)/new_sample_rate)
                this_trial_contours_X = []
                this_trial_contours_Y = []
                this_trial_contours = []
                this_trial_circles_X = []
                this_trial_circles_Y = []
                this_trial_circles = []
                # loop through the trial at given sample rate
                for sample in range(no_of_samples):
                    start = sample * new_sample_rate
                    end = (sample * new_sample_rate) + (new_sample_rate - 1)
                    this_slice = trial[start:end]
                    for line in this_slice:
                        if (line<0).any():
                            line[:] = np.nan
                        if (line>15000).any():
                            line[:] = np.nan
                    # extract pupil sizes and locations from valid time buckets
                    this_slice_contours_X = []
                    this_slice_contours_Y = []
                    this_slice_contours = []
                    this_slice_circles_X = []
                    this_slice_circles_Y = []
                    this_slice_circles = []
                    for frame in this_slice:
                        # contour x,y
                        ## DON'T PAIR X-Y YET
                        this_slice_contours_X.append(frame[0])
                        this_slice_contours_Y.append(frame[1])
                        # contour area
                        this_slice_contours.append(frame[2])
                        # circles x,y
                        ## DON'T PAIR X-Y YET
                        this_slice_circles_X.append(frame[3])
                        this_slice_circles_Y.append(frame[4])
                        # circles area
                        this_slice_circles.append(frame[5])
                    # average the pupil size and movement in this sample slice
                    this_slice_avg_contour_X = np.nanmean(this_slice_contours_X)
                    this_slice_avg_contour_Y = np.nanmean(this_slice_contours_Y)
                    this_slice_avg_contour = np.nanmean(this_slice_contours) 
                    this_slice_avg_circle_X = np.nanmean(this_slice_circles_X)
                    this_slice_avg_circle_Y = np.nanmean(this_slice_circles_Y)       
                    this_slice_avg_circle = np.nanmean(this_slice_circles)
                    # append to list of downsampled pupil sizes and movements
                    this_trial_contours_X.append(this_slice_avg_contour_X)
                    this_trial_contours_Y.append(this_slice_avg_contour_Y)
                    this_trial_contours.append(this_slice_avg_contour)
                    this_trial_circles_X.append(this_slice_avg_circle_X)
                    this_trial_circles_Y.append(this_slice_avg_circle_Y)
                    this_trial_circles.append(this_slice_avg_circle)
                # Find count of bad measurements
                bad_count_contours_X = sum(np.isnan(this_trial_contours_X))
                bad_count_contours_Y = sum(np.isnan(this_trial_contours_Y))
                bad_count_contours = sum(np.isnan(this_trial_contours))
                bad_count_circles_X = sum(np.isnan(this_trial_circles_X))
                bad_count_circles_Y = sum(np.isnan(this_trial_circles_Y))
                bad_count_circles = sum(np.isnan(this_trial_circles))
                # if more than half of the trial is NaN, then throw away this trial
                # otherwise, if it's a good enough trial...
                bad_threshold = no_of_samples/2
                if (bad_count_contours_X<bad_threshold): 
                    this_chunk_length = len(this_trial_contours_X)
                    data_contours_X[index][0:this_chunk_length] = this_trial_contours_X
                    data_contours_X[index][-1] = trial_stim_number
                if (bad_count_contours_Y<bad_threshold): 
                    this_chunk_length = len(this_trial_contours_Y)
                    data_contours_Y[index][0:this_chunk_length] = this_trial_contours_Y
                    data_contours_Y[index][-1] = trial_stim_number
                if (bad_count_contours<bad_threshold) or (bad_count_circles<bad_threshold): 
                    this_chunk_length = len(this_trial_contours)
                    data_contours[index][0:this_chunk_length] = this_trial_contours
                    data_contours[index][-1] = trial_stim_number
                if (bad_count_circles_X<bad_threshold): 
                    this_chunk_length = len(this_trial_circles_X)
                    data_circles_X[index][0:this_chunk_length] = this_trial_circles_X
                    data_circles_X[index][-1] = trial_stim_number
                if (bad_count_circles_Y<bad_threshold): 
                    this_chunk_length = len(this_trial_circles_Y)
                    data_circles_Y[index][0:this_chunk_length] = this_trial_circles_Y
                    data_circles_Y[index][-1] = trial_stim_number
                if (bad_count_circles<bad_threshold): 
                    this_chunk_length = len(this_trial_circles)
                    data_circles[index][0:this_chunk_length] = this_trial_circles
                    data_circles[index][-1] = trial_stim_number
                index = index + 1
            else:
                #print("Discarding trial {name}".format(name=trial_name))
                index = index + 1
                good_trials = good_trials - 1
        return data_contours_X, data_contours_Y, data_contours, data_circles_X, data_circles_Y, data_circles, num_trials, good_trials
    else: 
        print("Sample rate must be a multiple of {bucket}".format(bucket=original_bucket_size))

def threshold_to_nan(input_array, threshold, upper_or_lower):
    for index in range(len(input_array)): 
        if upper_or_lower=='upper':
            if np.isnan(input_array[index])==False and input_array[index]>threshold:
                input_array[index] = np.nan
        if upper_or_lower=='lower':
            if np.isnan(input_array[index])==False and input_array[index]<threshold:
                input_array[index] = np.nan
    return input_array

def filter_to_nan(list_of_dicts, upper_threshold, lower_threshold):
    for dictionary in list_of_dicts:
        for key in dictionary:
            for trial in dictionary[key]:
                trial = threshold_to_nan(trial, upper_threshold, 'upper')
                trial = threshold_to_nan(trial, lower_threshold, 'lower')
    return list_of_dicts

def find_windowed_peaks(time_bucket_dict, window, threshold):
    windowed_peaks = {}
    key_list = []
    for ptime in time_bucket_dict.keys():
        key_list.append(ptime)
    key_list.sort()
    for k,g in groupby(enumerate(key_list), lambda ix: ix[0] - ix[1]):
        consecutive_ptimes = list(map(itemgetter(1), g))
        #print(consecutive_ptimes)
        if len(consecutive_ptimes)<=window:
            max_val = threshold
            this_group_count = 0
            for time in consecutive_ptimes:
                this_group_count = this_group_count + time_bucket_dict[time]
            if this_group_count>max_val:
                max_time = np.median(consecutive_ptimes)
                windowed_peaks[int(max_time)] = this_group_count
        else:
            max_val = threshold
            max_times = {}
            for time in consecutive_ptimes:
                center = time
                start = int(center-(window/2))
                end = int(center+(window/2))
                this_group_count = 0
                for t in range(int(start),int(end)):
                    this_group_count = this_group_count + time_bucket_dict.get(t,0)
                if this_group_count>max_val:
                    if not max_times:
                        max_times[center] = this_group_count
                        max_val = this_group_count
                    else:
                        overlap = [x for x in max_times.keys() if start<x<end]
                        filtered_overlap = {}
                        for o in overlap:
                            temp_val = max_times.pop(o)
                            if temp_val>this_group_count:
                                filtered_overlap[o] = [temp_val]
                        if not filtered_overlap:
                            max_times[center] = this_group_count
                            max_val = this_group_count
                        else:
                            for f in filtered_overlap.items():
                                max_times[f[0]] = f[1]
            for max_time in max_times.keys():
                windowed_peaks[max_time] = max_times[max_time]
    return windowed_peaks

def calc_mvmnt_from_pos(list_of_positon_arrays, nans_threshold, movement_threshold_upper, movement_threshold_lower):
    this_stim_movements = []
    for trial in list_of_positon_arrays:
        trial_movements_min_len = len(trial)
        this_trial_movement = []
        nans_in_a_row = 0
        prev = np.nan
        for i in range(len(trial)):
            now = trial[i]
            #print("now: "+str(now))
            #print("prev: "+str(prev))
            if np.isnan(now):
                # keep the nan to understand where the dropped frames are
                this_trial_movement.append(np.nan)
                nans_in_a_row = nans_in_a_row + 1
                continue
            if nans_in_a_row>(nans_threshold):
                break
            if i==0:
                this_trial_movement.append(0)
                prev = now
                continue
            if not np.isnan(prev):
                movement = now - prev
                this_trial_movement.append(movement)
                prev = now
                nans_in_a_row = 0 
            #print("movements: " + str(this_trial_movement))
            #print("consecutive nans: " + str(nans_in_a_row))
        # filter out movements too large to be realistic saccades (120 pixels)
        trial_movement_array = np.array(this_trial_movement)
        trial_movement_array = threshold_to_nan(trial_movement_array, movement_threshold_upper, 'upper')
        trial_movement_array = threshold_to_nan(trial_movement_array, movement_threshold_lower, 'lower')
        this_stim_movements.append(trial_movement_array)  
    # filter for trial movements that are less than 4000 bins long
    output = [x for x in this_stim_movements if len(x)>=trial_movements_min_len]
    return output

def calc_avg_motion_and_peaks(list_of_movement_arrays, window):
    total_motion = np.zeros(len(list_of_movement_arrays[0]))
    nan_count = np.zeros(len(list_of_movement_arrays[0]))
    # for each frame, sum the abs(movements) on that frame
    for trial in list_of_movement_arrays:
        for t in range(len(trial)):
            if np.isnan(trial[t]):
                nan_count[t] = nan_count[t] + 1
            if not np.isnan(trial[t]):
                total_motion[t] = total_motion[t] + abs(trial[t])
    avg_motion = np.zeros(len(list_of_movement_arrays[0]))
    for f in range(len(total_motion)):
        valid_subjects_this_tbucket = len(list_of_movement_arrays) - nan_count[f]
        avg_motion[f] = total_motion[f]/valid_subjects_this_tbucket
    # smooth the average motion
    # smoothing window must be odd!
    # apply savitzky-golay filter to smooth
    avg_motion_smoothed = savgol_filter(avg_motion, window, 3)
    # find peaks in average motion
    peaks, _ = find_peaks(avg_motion_smoothed, height=(2,10), prominence=0.75)
    return avg_motion_smoothed, peaks

def find_saccades(list_of_movement_arrays, saccade_threshold, raw_count_threshold, window_size, windowed_count_threshold):
    all_trials_peaks = []
    for trial in range(len(list_of_movement_arrays)):
        all_trials_peaks.append([])
        this_trial = list_of_movement_arrays[trial]
        for time_bucket in range(len(this_trial)):
            # find timebuckets where abs(movement)>threshold
            if abs(this_trial[time_bucket])>=saccade_threshold:
                all_trials_peaks[trial].append(time_bucket)
    # count number of subjects who had peaks in the same timebuckets
    trial_peaks_totals = {}
    trial_peaks_totals = defaultdict(lambda:0, trial_peaks_totals)
    for trial in all_trials_peaks:
        for tbucket in trial:
            trial_peaks_totals[tbucket] = trial_peaks_totals[tbucket] + 1
    # filter for timebuckets when "enough" subjects had peaks
    peak_tbuckets_filtered = {}
    # combine counts of peaks within time windows
    for key in trial_peaks_totals.keys():
        count = trial_peaks_totals[key]
        if count>=raw_count_threshold:
            #print(t, count)
            peak_tbuckets_filtered[key] = count
    # combine counts of peaks within time windows
    peak_tbuckets_windowed = find_windowed_peaks(peak_tbuckets_filtered, window_size, windowed_count_threshold)
    saccades = {tbucket:total for tbucket,total in peak_tbuckets_windowed.items()}
    return saccades

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

def write_avg_world_vid(avg_world_vid_tbucketed_dict, start_tbucket, end_tbucket, write_path):
    # temporarily switch matplotlib backend in order to write video
    plt.switch_backend("Agg")
    # convert dictionary of avg world vid frames into a list of arrays
    tbucket_frames = []
    sorted_tbuckets = sorted([x for x in avg_world_vid_tbucketed_dict.keys() if type(x) is int])
    for tbucket in sorted_tbuckets:
        tbucket_frames.append(avg_world_vid_tbucketed_dict[tbucket])
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    FF_writer = animation.FFMpegWriter(fps=30, codec='h264', metadata=dict(artist='Danbee Kim'))
    fig = plt.figure()
    i = start_tbucket
    im = plt.imshow(tbucket_frames[i], cmap='gray', animated=True)
    def updatefig(*args):
        global i
        if (i<end_tbucket):
            i += 1
        else:
            i=0
        im.set_array(tbucket_frames[i])
        return im,
    ani = animation.FuncAnimation(fig, updatefig, frames=len(tbucket_frames), interval=50, blit=True)
    print("Writing average world video frames to {path}...".format(path=write_path))
    ani.save(write_path, writer=FF_writer)
    plt.close(fig)
    print("Finished writing!")
    # restore default matplotlib backend
    plt.switch_backend('TkAgg')
       
def display_avg_world_tbucket(avg_world_vid_tbucketed_dict, tbucket_to_display):
    sorted_tbuckets = sorted([x for x in avg_world_vid_tbucketed_dict.keys() if type(x) is int])
    max_tbucket = sorted_tbuckets[-1]
    if 0<=tbucket_to_display<=max_tbucket:
        display = avg_world_vid_tbucketed_dict[tbucket_to_display]
        fig = plt.figure()
        im = plt.imshow(display, cmap='gray')
        plt.show()
    else: 
        print("Time Bucket is out of range!")

def find_moment_tbuckets(list_of_moments_to_find, all_moments_dict, year_month, this_stimulus):
    this_year_int = int(year_month.split('-')[0])
    this_month_int = int(year_month.split('-')[1])
    tbuckets_of_moments = {}
    for m in list_of_moments_to_find:
        if len(all_moments_dict[this_stimulus][m].keys())>1:
            for tbucket_num in all_moments_dict[this_stimulus][m].keys():
                if this_month_int in all_moments_dict[this_stimulus][m][tbucket_num]:
                    tbuckets_of_moments[m] = tbucket_num
        else:
            for tbucket_num in all_moments_dict[this_stimulus][m].keys():
                tbuckets_of_moments[m] = tbucket_num
    # if a month has not been checked for moments of interest, find the nearest month that has been checked
    if len(tbuckets_of_moments)!=len(list_of_moments_to_find):
        for m in list_of_moments_to_find:
            if m not in tbuckets_of_moments:
                nearest_year_month = None
                nearest_year_diff = math.inf
                nearest_month_diff = math.inf
                for tbucket_num in all_moments_dict[this_stimulus][m].keys():
                    for year_month_str in all_moments_dict[this_stimulus][m][tbucket_num]:
                        year_int = int(year_month_str.split('-')[0])
                        month_int = int(year_month_str.split('-')[1])
                        this_year_diff = abs(this_year_int-year_int)
                        this_month_diff = abs(this_month_int-month_int)
                        if this_year_diff<nearest_year_diff and this_month_diff<nearest_month_diff:
                            nearest_year_diff = this_year_diff
                            nearest_month_diff = this_month_diff
                            nearest_year_month = year_month_str
                for tbucket_num in all_moments_dict[this_stimulus][m].keys():
                    if nearest_year_month in all_moments_dict[this_stimulus][m][tbucket_num]:
                        tbuckets_of_moments[m] = tbucket_num
    return tbuckets_of_moments

def pooled_for_global_moment_of_interest(full_avg_world_vids_dict, all_moments_dict, moment_start_str, moment_end_str):
    all_pooled_tbuckets = {}
    monthly_pooled_avg_tbuckets = {}
    for month in full_avg_world_vids_dict.keys():
        monthly_pooled_avg_tbuckets[month] = {}
        total_vid_count = 0
        start_tbuckets = {}
        end_tbuckets = {}
        all_stims_summed = {}
        for stim in full_avg_world_vids_dict[month].keys():
            all_stims_summed[stim] = {}
            total_vid_count = total_vid_count + full_avg_world_vids_dict[month][stim]['Vid Count']
            v_height = full_avg_world_vids_dict[month][stim]['Vid Dimensions'][0]
            v_width = full_avg_world_vids_dict[month][stim]['Vid Dimensions'][1]
            empty_frame = np.zeros((v_height,v_width))
            # find start and end of this moment
            moments_to_find = [moment_start_str, moment_end_str]
            moments_tbuckets = find_moment_tbuckets(moments_to_find, all_moments_dict, month, stim)
            start_tbucket = moments_tbuckets[moment_start_str] - 1
            end_tbucket = moments_tbuckets[moment_end_str] + 1
            start_tbuckets[stim] = start_tbucket
            end_tbuckets[stim] = end_tbucket
            len_this_moment_tbuckets = end_tbucket - start_tbucket
            for bucket in range(len_this_moment_tbuckets):
                all_stims_summed[stim][bucket] = {}
            ordered_tbuckets = sorted([tbucket for tbucket in full_avg_world_vids_dict[month][stim].keys() if type(tbucket) is int])
            this_moment_frames = ordered_tbuckets[start_tbucket:end_tbucket]
            for t in range(len(this_moment_frames)):
                if not np.any(np.isnan(full_avg_world_vids_dict[month][stim][this_moment_frames[t]])):
                    all_stims_summed[stim][t]['Summed Frame'] = all_stims_summed[stim][t].get('Summed Frame',empty_frame) + full_avg_world_vids_dict[month][stim][this_moment_frames[t]]
                    all_stims_summed[stim][t]['Frame Count'] = all_stims_summed[stim][t].get('Frame Count',0) + 1
        print("Building average frames for month {m}, moment of interest: {s} to {e}".format(m=month,s=moment_start_str,e=moment_end_str))
        this_month_avg_tbuckets = {}
        for stim in all_stims_summed.keys():
            for tbucket in all_stims_summed[stim].keys():
                this_month_avg_tbuckets[tbucket] = {}
        for stim in all_stims_summed.keys():
            for tbucket in all_stims_summed[stim].keys():
                if 'Summed Frame' in all_stims_summed[stim][tbucket].keys():
                    this_month_avg_tbuckets[tbucket]['Summed Frame'] = this_month_avg_tbuckets[tbucket].get('Summed Frame',empty_frame) + all_stims_summed[stim][tbucket]['Summed Frame']
                    this_month_avg_tbuckets[tbucket]['Frame Count'] = this_month_avg_tbuckets[tbucket].get('Frame Count',0) + 1
        for tbucket in this_month_avg_tbuckets.keys():
            if 'Summed Frame' in this_month_avg_tbuckets[tbucket].keys():
                monthly_pooled_avg_tbuckets[month][tbucket] = this_month_avg_tbuckets[tbucket]['Summed Frame']/float(this_month_avg_tbuckets[tbucket]['Frame Count'])
        monthly_pooled_avg_tbuckets[month]['Vid Count'] = total_vid_count
        monthly_pooled_avg_tbuckets[month]['starts'] = start_tbuckets
        monthly_pooled_avg_tbuckets[month]['ends'] = end_tbuckets
    print("Averaging across all months...")
    all_months = {}
    total_vids = 0
    all_stim_starts = {stim:{} for stim in full_avg_world_vids_dict[month].keys()}
    all_stim_ends = {stim:{} for stim in full_avg_world_vids_dict[month].keys()}
    for month in monthly_pooled_avg_tbuckets.keys():
        for tbucket in monthly_pooled_avg_tbuckets[month].keys():
            all_months[tbucket] = {}
    for month in monthly_pooled_avg_tbuckets.keys():
        for tbucket in monthly_pooled_avg_tbuckets[month].keys():
            if tbucket=='Vid Count':
                total_vids = total_vids + monthly_pooled_avg_tbuckets[month]['Vid Count']
                continue
            if tbucket=='starts':
                for stim in monthly_pooled_avg_tbuckets[month]['starts']:
                    all_stim_starts[stim][month] = monthly_pooled_avg_tbuckets[month]['starts'][stim]
                continue
            if tbucket=='ends':
                for stim in monthly_pooled_avg_tbuckets[month]['ends']:
                    all_stim_ends[stim][month] = monthly_pooled_avg_tbuckets[month]['ends'][stim]
                continue
            if type(tbucket) is int:
                all_months[tbucket]['Summed Frame'] = all_months[tbucket].get('Summed Frame',empty_frame) + monthly_pooled_avg_tbuckets[month][tbucket]
                all_months[tbucket]['Frame Count'] = all_months[tbucket].get('Frame Count',0) + 1
    for tbucket in all_months.keys():
        all_pooled_tbuckets[tbucket] = {}
    for tbucket in all_months.keys():
        if 'Summed Frame' in all_months[tbucket].keys():
            all_pooled_tbuckets[tbucket] = all_months[tbucket]['Summed Frame']/float(all_months[tbucket]['Frame Count'])
    all_pooled_tbuckets['Vid Count'] = total_vids
    all_pooled_tbuckets['starts'] = all_stim_starts
    all_pooled_tbuckets['ends'] = all_stim_ends
    return all_pooled_tbuckets

def pooled_for_stim_specific_moment_of_interest(full_avg_world_vids_dict, all_stims_list, all_moments_dict, moment_start_str, moment_end_str):
    stims_pooled_avg_tbuckets = {}
    for stimulus in all_stims_list:
        stims_pooled_avg_tbuckets[stimulus] = {}
        this_stim_all_months = {}
        total_vid_count = 0
        start_tbuckets = {}
        end_tbuckets = {}
        for month in full_avg_world_vids_dict.keys():
            start_tbuckets[month] = {}
            end_tbuckets[month] = {}
            # find start and end of this moment
            moments_to_find = [moment_start_str, moment_end_str]
            moments_tbuckets = find_moment_tbuckets(moments_to_find, all_moments_dict, month, stimulus)
            start_tbucket = moments_tbuckets[moment_start_str] - 1
            end_tbucket = moments_tbuckets[moment_end_str] + 1
            start_tbuckets[month] = start_tbucket
            end_tbuckets[month] = end_tbucket
            len_of_this_moment_tbuckets = end_tbucket - start_tbucket
            for bucket in range(len_of_this_moment_tbuckets):
                this_stim_all_months[bucket] = {}
        for month in full_avg_world_vids_dict.keys():
            total_vid_count = total_vid_count + full_avg_world_vids_dict[month][stimulus]['Vid Count']
            v_height = full_avg_world_vids_dict[month][stimulus]['Vid Dimensions'][0]
            v_width = full_avg_world_vids_dict[month][stimulus]['Vid Dimensions'][1]
            empty_frame = np.zeros((v_height,v_width))
            ordered_tbuckets = sorted([tbucket for tbucket in full_avg_world_vids_dict[month][stim].keys() if type(tbucket) is int])
            this_moment_frames = ordered_tbuckets[start_tbucket:end_tbucket]
            for t in range(len(this_moment_frames)):
                if not np.any(np.isnan(full_avg_world_vids_dict[month][stimulus][this_moment_frames[t]])):
                    this_stim_all_months[t]['Summed Frame'] = this_stim_all_months[t].get('Summed Frame',empty_frame) + full_avg_world_vids_dict[month][stimulus][this_moment_frames[t]]
                    this_stim_all_months[t]['Frame Count'] = this_stim_all_months[t].get('Frame Count',0) + 1
        print("Building average frames for stimulus {stim}, moment of interest: {s} to {e}".format(stim=stimulus,s=moment_start_str,e=moment_end_str))
        for tbucket in this_stim_all_months.keys():
            if type(tbucket) is int:
                stims_pooled_avg_tbuckets[stimulus][tbucket] = {}
        for tbucket in this_stim_all_months.keys():
            if 'Summed Frame' in this_stim_all_months[tbucket].keys():
                stims_pooled_avg_tbuckets[stimulus][tbucket] = this_stim_all_months[tbucket]['Summed Frame']/float(this_stim_all_months[tbucket]['Frame Count'])
        stims_pooled_avg_tbuckets[stimulus]['Vid Count'] = total_vid_count
        stims_pooled_avg_tbuckets[stimulus]['starts'] = start_tbuckets
        stims_pooled_avg_tbuckets[stimulus]['ends'] = end_tbuckets
    return stims_pooled_avg_tbuckets

def smoothed_baselined_lum_of_tb_world_vid(dict_of_avg_world_vid_tbuckets, smoothing_window_size, baseline_no_tbuckets):
    total_vids = dict_of_avg_world_vid_tbuckets['Vid Count']
    luminances_list = []
    sorted_tbuckets = sorted([x for x in dict_of_avg_world_vid_tbuckets.keys() if type(x) is int])
    for bucket in sorted_tbuckets:
        if type(dict_of_avg_world_vid_tbuckets[bucket]) is not dict:
            this_tbucket_luminance = np.nansum(dict_of_avg_world_vid_tbuckets[bucket])
        else:
            this_tbucket_luminance = np.nan
        luminances_list.append(this_tbucket_luminance)
    luminances_array = np.array(luminances_list)
    smoothed_lum_array = savgol_filter(luminances_array, smoothing_window_size, 3)
    baseline_this_vid = np.nanmean(smoothed_lum_array[0:baseline_no_tbuckets])
    smoothed_baselined_lum_array = [(float(x-baseline_this_vid)/baseline_this_vid) for x in smoothed_lum_array]
    return np.array(smoothed_baselined_lum_array)

def pool_pupil_data_for_global_moment_of_interest(pupil_data_dict, moment_of_interest_pooled_world_vid_dict, pooled_pupil_data):
    for stim in pupil_data_dict.keys():
        this_stim_all_months_moment_starts = []
        this_stim_all_months_moment_ends = []
        start_end_collected = [this_stim_all_months_moment_starts, this_stim_all_months_moment_ends]
        start_end_keys = ['starts', 'ends']
        for x in range(len(start_end_keys)):
            for month in moment_of_interest_pooled_world_vid_dict[start_end_keys[x]][stim].keys():
                start_end_collected[x].append(moment_of_interest_pooled_world_vid_dict[start_end_keys[x]][stim][month])
        this_stim_moment_start = min(this_stim_all_months_moment_starts)
        this_stim_moment_end = max(this_stim_all_months_moment_ends)
        for trial in range(len(pupil_data_dict[stim])):
            pooled_pupil_data.append(np.array(pupil_data_dict[stim][trial][this_stim_moment_start:this_stim_moment_end]))

def pool_pupil_means_for_global_moment_of_interest(pupil_means_dict, moment_of_interest_pooled_world_vid_dict, pooled_pupil_means):
    for stim in pupil_means_dict.keys():
        this_stim_all_months_moment_starts = []
        this_stim_all_months_moment_ends = []
        start_end_collected = [this_stim_all_months_moment_starts, this_stim_all_months_moment_ends]
        start_end_keys = ['starts', 'ends']
        for x in range(len(start_end_keys)):
            for month in moment_of_interest_pooled_world_vid_dict[start_end_keys[x]][stim].keys():
                start_end_collected[x].append(moment_of_interest_pooled_world_vid_dict[start_end_keys[x]][stim][month])
        this_stim_moment_start = min(this_stim_all_months_moment_starts)
        this_stim_moment_end = max(this_stim_all_months_moment_ends)
        pooled_pupil_means.append(np.array(pupil_means_dict[stim][this_stim_moment_start:this_stim_moment_end]))

def pool_pupil_peaks_for_global_moment_of_interest(pupil_peaks_dict, moment_of_interest_pooled_world_vid_dict, pooled_pupil_peaks):
    for stim in pupil_peaks_dict.keys():
        this_stim_all_months_moment_starts = []
        this_stim_all_months_moment_ends = []
        start_end_collected = [this_stim_all_months_moment_starts, this_stim_all_months_moment_ends]
        start_end_keys = ['starts', 'ends']
        for x in range(len(start_end_keys)):
            for month in moment_of_interest_pooled_world_vid_dict[start_end_keys[x]][stim].keys():
                start_end_collected[x].append(moment_of_interest_pooled_world_vid_dict[start_end_keys[x]][stim][month])
        this_stim_moment_start = min(this_stim_all_months_moment_starts)
        this_stim_moment_end = max(this_stim_all_months_moment_ends)
        for peak in pupil_peaks_dict[stim]:
            if this_stim_moment_start<=peak<=this_stim_moment_end:
                pooled_pupil_peaks.append(peak)

def pool_pupil_saccades_for_global_moment_of_interest(pupil_saccades_dict, moment_of_interest_pooled_world_vid_dict, pooled_pupil_saccades_dict):
    for stim in pupil_saccades_dict.keys():
        this_stim_all_months_moment_starts = []
        this_stim_all_months_moment_ends = []
        start_end_collected = [this_stim_all_months_moment_starts, this_stim_all_months_moment_ends]
        start_end_keys = ['starts', 'ends']
        for x in range(len(start_end_keys)):
            for month in moment_of_interest_pooled_world_vid_dict[start_end_keys[x]][stim].keys():
                start_end_collected[x].append(moment_of_interest_pooled_world_vid_dict[start_end_keys[x]][stim][month])
        this_stim_moment_start = min(this_stim_all_months_moment_starts)
        this_stim_moment_end = max(this_stim_all_months_moment_ends)
        for threshold in pupil_saccades_dict[stim]:
            for s_tbucket in pupil_saccades_dict[stim][threshold]:
                if this_stim_moment_start<=s_tbucket<=this_stim_moment_end:
                    pooled_pupil_saccades_dict[threshold][s_tbucket] = pooled_pupil_saccades_dict[threshold].get(s_tbucket,0) + pupil_saccades_dict[stim][threshold][s_tbucket]

# set up log file to store all printed messages
current_working_directory = os.getcwd()
class Logger(object):
    def __init__(self):
        # grab today's date
        now = datetime.datetime.now()
        log_filename = "data-extraction_log_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
        log_file = os.path.join(current_working_directory, log_filename)
        self.terminal = sys.stdout
        self.log = open(log_file, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  
    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    
sys.stdout = Logger()
### BEGIN ANALYSIS ###
# List relevant data locations: these are for KAMPFF-LAB-VIDEO
#root_folder = r"C:\Users\KAMPFF-LAB-VIDEO\Dropbox\SurprisingMinds\analysis\pythonWithAdam-csv"
root_folder = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\pythonWithAdam-csv"
# set up folders
plots_folder = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\plots"
pupils_folder = os.path.join(plots_folder, "pupil")
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
if not os.path.exists(engagement_folder):
    #print("Creating engagement count folder.")
    os.makedirs(engagement_folder)
if not os.path.exists(linReg_folder):
    #print("Creating engagement count folder.")
    os.makedirs(linReg_folder)
if not os.path.exists(pooled_stim_vids_folder):
    #print("Creating engagement count folder.")
    os.makedirs(pooled_stim_vids_folder)
# consolidate csv files from multiple days into one data structure
day_folders = sorted(os.listdir(root_folder))
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
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
### BEGIN PUPIL DATA EXTRACTION ###
# prepare to sort pupil data by stimulus
all_right_trials_contours_X = {key:[] for key in stim_vids}
all_right_trials_contours_Y = {key:[] for key in stim_vids}
all_right_trials_contours = {key:[] for key in stim_vids}
all_right_trials_circles_X = {key:[] for key in stim_vids}
all_right_trials_circles_Y = {key:[] for key in stim_vids}
all_right_trials_circles = {key:[] for key in stim_vids}
all_left_trials_contours_X = {key:[] for key in stim_vids}
all_left_trials_contours_Y = {key:[] for key in stim_vids}
all_left_trials_contours = {key:[] for key in stim_vids}
all_left_trials_circles_X = {key:[] for key in stim_vids}
all_left_trials_circles_Y = {key:[] for key in stim_vids}
all_left_trials_circles = {key:[] for key in stim_vids}
all_trials_position_X_data = [all_right_trials_contours_X, all_right_trials_circles_X, all_left_trials_contours_X, all_left_trials_circles_X]
all_trials_position_Y_data = [all_right_trials_contours_Y, all_right_trials_circles_Y, all_left_trials_contours_Y, all_left_trials_circles_Y]
all_trials_size_data = [all_right_trials_contours, all_right_trials_circles, all_left_trials_contours, all_left_trials_circles]
activation_count = []
analysed_count = []
stimuli_tbucketed = {key:[] for key in stim_vids}
# find pupil data on dropbox
pupil_folders = fnmatch.filter(day_folders, 'SurprisingMinds_*')
# first day was a debugging session, so skip it
pupil_folders = pupil_folders[1:]
# currently still running pupil finding analysis...
pupil_folders = pupil_folders[:-1]
for day_folder in pupil_folders: 
    # for each day...
    day_folder_path = os.path.join(root_folder, day_folder)
    analysis_folder = os.path.join(day_folder_path, "Analysis")
    csv_folder = os.path.join(analysis_folder, "csv")
    world_folder = os.path.join(analysis_folder, "world")

    # Print/save number of users per day
    day_name = day_folder.split("_")[-1]
    try: 
        ## EXTRACT PUPIL SIZE AND POSITION
        right_area_contours_X, right_area_contours_Y, right_area_contours, right_area_circles_X, right_area_circles_Y, right_area_circles, num_right_activations, num_good_right_trials = load_daily_pupils("right", csv_folder, downsampled_no_of_time_buckets, original_bucket_size_in_ms, downsampled_bucket_size_ms)
        left_area_contours_X, left_area_contours_Y, left_area_contours, left_area_circles_X, left_area_circles_Y, left_area_circles, num_left_activations, num_good_left_trials = load_daily_pupils("left", csv_folder, downsampled_no_of_time_buckets, original_bucket_size_in_ms, downsampled_bucket_size_ms)

        analysed_count.append((num_good_right_trials, num_good_left_trials))
        activation_count.append((num_right_activations, num_left_activations))
        print("On {day}, exhibit was activated {right_count} times (right) and {left_count} times (left), with {right_good_count} good right trials and {left_good_count} good left trials".format(day=day_name, right_count=num_right_activations, left_count=num_left_activations, right_good_count=num_good_right_trials, left_good_count=num_good_left_trials))

        # separate by stimulus number
        R_contours_X = {key:[] for key in stim_vids}
        R_contours_Y = {key:[] for key in stim_vids}
        R_contours = {key:[] for key in stim_vids}
        R_circles_X = {key:[] for key in stim_vids}
        R_circles_Y = {key:[] for key in stim_vids}
        R_circles = {key:[] for key in stim_vids}
        L_contours_X = {key:[] for key in stim_vids}
        L_contours_Y = {key:[] for key in stim_vids}
        L_contours = {key:[] for key in stim_vids}
        L_circles_X = {key:[] for key in stim_vids}
        L_circles_Y = {key:[] for key in stim_vids}
        L_circles = {key:[] for key in stim_vids}

        stim_sorted_data_right = [R_contours_X, R_contours_Y, R_contours, R_circles_X, R_circles_Y, R_circles]
        stim_sorted_data_left = [L_contours_X, L_contours_Y, L_contours, L_circles_X, L_circles_Y, L_circles]
        stim_sorted_data_all = [stim_sorted_data_right, stim_sorted_data_left]

        extracted_data_right = [right_area_contours_X, right_area_contours_Y, right_area_contours, right_area_circles_X, right_area_circles_Y, right_area_circles]
        extracted_data_left = [left_area_contours_X, left_area_contours_Y, left_area_contours, left_area_circles_X, left_area_circles_Y, left_area_circles]
        extracted_data_all = [extracted_data_right, extracted_data_left]

        for side in range(len(extracted_data_all)):
            for dataset in range(len(extracted_data_all[side])):
                for trial in extracted_data_all[side][dataset]:
                    stim_num = trial[-1]
                    if stim_num in stim_sorted_data_all[side][dataset].keys():
                        stim_sorted_data_all[side][dataset][stim_num].append(trial[:-1])

        # filter data for outlier points
        all_position_X_data = [R_contours_X, R_circles_X, L_contours_X, L_circles_X]
        all_position_Y_data = [R_contours_Y, R_circles_Y, L_contours_Y, L_circles_Y]
        all_size_data = [R_contours, R_circles, L_contours, L_circles]
        # remove:
        # eye positions that are not realistic
        # time buckets with no corresponding frames
        # video pixel limits are (798,599)
        all_position_X_data = filter_to_nan(all_position_X_data, 798, 0)
        all_position_Y_data = filter_to_nan(all_position_Y_data, 599, 0)
        # contours/circles that are too big
        all_size_data = filter_to_nan(all_size_data, 15000, 0)

        # create a baseline for size data
        R_contours_baseline = {key:[] for key in stim_vids}
        R_circles_baseline = {key:[] for key in stim_vids}
        L_contours_baseline = {key:[] for key in stim_vids}
        L_circles_baseline = {key:[] for key in stim_vids}
        all_size_baselines = [R_contours_baseline, R_circles_baseline, L_contours_baseline, L_circles_baseline]

        for dataset in range(len(all_size_data)):
            for stimulus in all_size_data[dataset]: 
                for trial in all_size_data[dataset][stimulus]:
                    baseline = np.nanmedian(trial[:baseline_no_buckets])
                    all_size_baselines[dataset][stimulus].append(baseline)

        # append position data to global data structure
        for i in range(len(all_position_X_data)):
            for stimulus in all_position_X_data[i]:
                for index in range(len(all_position_X_data[i][stimulus])):
                    all_trials_position_X_data[i][stimulus].append(all_position_X_data[i][stimulus][index])
        for i in range(len(all_position_Y_data)):
            for stimulus in all_position_Y_data[i]:
                for index in range(len(all_position_Y_data[i][stimulus])):
                    all_trials_position_Y_data[i][stimulus].append(all_position_Y_data[i][stimulus][index])
        # normalize and append size data to global data structure
        for i in range(len(all_size_data)):
            for stimulus in all_size_data[i]:
                for index in range(len(all_size_data[i][stimulus])):
                    all_size_data[i][stimulus][index] = (all_size_data[i][stimulus][index]-all_size_baselines[i][stimulus][index])/all_size_baselines[i][stimulus][index]
                    all_trials_size_data[i][stimulus].append(all_size_data[i][stimulus][index])
        print("Day {day} succeeded!".format(day=day_name))
    except Exception:
        print("Day {day} failed!".format(day=day_name))
### END PUPIL EXTRACTION ###
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
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
    # save 
    avg_files_this_month = os.listdir(month_folder_path)
    vids_already_made = fnmatch.filter(avg_files_this_month, '*.mp4')
    for stim in downsampled_monthly_world_vids.keys():
        # save to all_months_avg_world_vids
        all_months_avg_world_vids[month_name][stim] = {}
        for key in downsampled_monthly_world_vids[stim].keys():
            all_months_avg_world_vids[month_name][stim][key] = downsampled_monthly_world_vids[stim][key]
        # write average world video to file
        stim_name = stim_float_to_name[stim]
        search_pattern = month_name + '_' + stim_name + '_AvgWorldVid*.mp4'
        this_stim_vid_already_made = fnmatch.filter(vids_already_made, search_pattern)
        if not this_stim_vid_already_made:
            tbuckets = sorted([x for x in downsampled_monthly_world_vids[stim].keys() if type(x) is int])
            start_bucket = 0
            end_bucket = tbuckets[-1]
            i = start_bucket
            num_vids_in_avg_vid = downsampled_monthly_world_vids[stim]['Vid Count']
            write_filename = str(month_name) + '_' + stim_name + '_AvgWorldVid' + str(num_vids_in_avg_vid) + '.mp4'
            write_path = os.path.join(month_folder_path, write_filename)
            write_avg_world_vid(downsampled_monthly_world_vids[stim], start_bucket, end_bucket, write_path)
        else:
            print("Monthly averaged stimulus videos already made for stimulus {s}".format(s=stim))
### END MONTHLY AVERAGE DATA EXTRACTION ###
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
### CSV DATA EXTRACTION COMPLETE ###
### GLOBAL VARIABLES FOR CLEANING AND PROCESSING EXTRACTED DATA ###
smoothing_window = 25 # in time buckets, must be odd! for savgol_filter
todays_datetime = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
### FIND MOMENTS OF INTEREST IN AVERAGE WORLD VIDS
# setup 
moments_of_interest = ['calibration start', 'calibration end', 'unique start', 'unique end', 'octo start', 'octo appears', 'octo end']
all_avg_world_moments = {key:{} for key in stim_vids}
for stim in all_avg_world_moments.keys():
    all_avg_world_moments[stim] = {moment:{} for moment in moments_of_interest}
# ------------------------------------------------------------------------ #
### MANUAL SECTION, UNCOMMENT BELOW TO FIND TIME BUCKETS OF MOMENTS OF INTEREST ###
# ------------------------------------------------------------------------ #
""" # start searching for time bucket numbers
# display available months
months_available = [x for x in all_months_avg_world_vids.keys()]
print("Months for which averaged stimulus video data exists: ")
for i in range(len(months_available)):
    print("{index}: {month}".format(index=i, month=months_available[i]))
# change the following variables based on what month/stim you want to check
### month/stimulus variables to change ###
month_index = 1 # change index to change month
stim_to_check = 29.0 # stims = 24.0, 25.0, 26.0, 27.0, 28.0, 29.0
# more setup
month_to_check = months_available[month_index]
avg_month_vid_dict_to_check = all_months_avg_world_vids[month_to_check][stim_to_check]
sorted_tbuckets = sorted([x for x in avg_month_vid_dict_to_check.keys() if type(x) is int])
max_tbucket = sorted_tbuckets[-1]
print("Time bucket to check must be smaller than {m}".format(m=max_tbucket))
### tbucket variable to change ###
tbucket_to_check = 1111 # change to check different time buckets
display_avg_world_tbucket(avg_month_vid_dict_to_check, tbucket_to_check) """
# ------------------------------------------------------------------------ #
### END MANUAL SECTION ###
# ------------------------------------------------------------------------ #
### RERUN THIS SECTION TO UPDATE MOMENTS OF INTEREST
### once found, manually insert time bucket numbers for moments of interest
### format --> for 'start' and 'appears' moments: 
### {first tbucket when a frame of this moment is visible in the monthly avg frame: [list of months for which this applies]}
### for 'end' moments:
### {last tbucket when a frame of this moment is visible in the monthly avg frame: [list of months for which this applies]}
# Stimulus 24.0
all_avg_world_moments[24.0] = {'calibration start': {102:['2017-10','2017-11']}, 
'calibration end': {441:['2017-10','2017-11']}, 
'unique start': {442:['2017-10'],443:['2017-11']}, 
'unique end': {596:['2017-10','2017-11']}, 
'octo start': {595:['2017-10'],596:['2017-11']}, 
'octo appears': {759:['2017-10','2017-11']}, 
'octo end': {987:['2017-10'],989:['2017-11']}} 
# Stimulus 25.0
all_avg_world_moments[25.0] = {'calibration start': {102:['2017-10','2017-11']}, 
'calibration end': {441:['2017-10','2017-11']}, 
'unique start': {443:['2017-10','2017-11']}, 
'unique end': {599:['2017-10'],600:['2017-11']}, 
'octo start': {599:['2017-10','2017-11']}, 
'octo appears': {763:['2017-10','2017-11']}, 
'octo end': {989:['2017-10'],993:['2017-11']}} 
# Stimulus 26.0
all_avg_world_moments[26.0] = {'calibration start': {102:['2017-10','2017-11']}, 
'calibration end': {441:['2017-10','2017-11']}, 
'unique start': {442:['2017-10'],443:['2017-11']}, 
'unique end': {663:['2017-10'],665:['2017-11']}, 
'octo start': {662:['2017-10'],664:['2017-11']}, 
'octo appears': {827:['2017-10','2017-11']}, 
'octo end': {1054:['2017-10'],1059:['2017-11']}} 
# Stimulus 27.0
all_avg_world_moments[27.0] = {'calibration start': {102:['2017-10','2017-11']}, 
'calibration end': {441:['2017-10','2017-11']}, 
'unique start': {443:['2017-10','2017-11']}, 
'unique end': {606:['2017-10'],607:['2017-11']}, 
'octo start': {605:['2017-10','2017-11']}, 
'octo appears': {770:['2017-10'],772:['2017-11']}, 
'octo end': {996:['2017-10'],1000:['2017-11']}} 
# Stimulus 28.0
all_avg_world_moments[28.0] = {'calibration start': {102:['2017-10','2017-11']}, 
'calibration end': {441:['2017-10','2017-11']}, 
'unique start': {443:['2017-10','2017-11']}, 
'unique end': {662:['2017-10'],663:['2017-11']}, 
'octo start': {661:['2017-10'],663:['2017-11']}, 
'octo appears': {826:['2017-10','2017-11']}, 
'octo end': {1054:['2017-10'],1056:['2017-11']}} 
# Stimulus 29.0
all_avg_world_moments[29.0] = {'calibration start': {102:['2017-10','2017-11']}, 
'calibration end': {441:['2017-10','2017-11']}, 
'unique start': {442:['2017-10'],443:['2017-11']}, 
'unique end': {717:['2017-10','2017-11']}, 
'octo start': {716:['2017-10'],717:['2017-11']}, 
'octo appears': {881:['2017-10'],882:['2017-11']}, 
'octo end': {1108:['2017-10'],1110:['2017-11']}} 
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
### ------------------------------ ###
### POOL AVG TBUCKETS ACROSS CALBIRATION SEQUENCE
all_cal_avg_tbuckets = pooled_for_global_moment_of_interest(all_months_avg_world_vids, all_avg_world_moments, 'calibration start', 'calibration end')
### ------------------------------ ###
### POOL AVG TBUCKETS ACROSS OCTOPUS CLIP
all_octo_avg_tbuckets = pooled_for_global_moment_of_interest(all_months_avg_world_vids, all_avg_world_moments, 'octo start', 'octo end')
### ------------------------------ ###
### POOL AVG TBUCKETS ACROSS EACH STIMULUS UNIQUE CLIP
all_stims_unique_clip_avg_tbuckets = pooled_for_stim_specific_moment_of_interest(all_months_avg_world_vids, stim_vids, all_avg_world_moments, 'unique start', 'unique end')
### ------------------------------ ###
### double check that the pooled avg videos make sense
display_avg_world_tbucket(all_octo_avg_tbuckets, 300)
### PREPARE FOR PLOTTING
all_cal_avg_lum_smoothed_baselined = smoothed_baselined_lum_of_tb_world_vid(all_cal_avg_tbuckets, smoothing_window, baseline_no_buckets)
all_octo_avg_lum_smoothed_baselined = smoothed_baselined_lum_of_tb_world_vid(all_octo_avg_tbuckets, smoothing_window, baseline_no_buckets)
all_stims_unique_avg_lum_smoothed_baselined = {}
for stim in all_stims_unique_clip_avg_tbuckets.keys():
    all_stims_unique_avg_lum_smoothed_baselined[stim] = smoothed_baselined_lum_of_tb_world_vid(all_stims_unique_clip_avg_tbuckets[stim], smoothing_window, baseline_no_buckets)
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
### EXHIBIT ACTIVITY METADATA ### 
# Save activation count to csv
engagement_count_filename = 'Exhibit_Activation_Count_measured-' + todays_datetime + '.csv'
engagement_data_folder = os.path.join(current_working_directory, 'Exhibit-Engagement')
if not os.path.exists(engagement_data_folder):
    #print("Creating plots folder.")
    os.makedirs(engagement_data_folder)
csv_file = os.path.join(engagement_data_folder, engagement_count_filename)
np.savetxt(csv_file, activation_count, fmt='%.2f', delimiter=',')
# activation count
total_activation = sum(count[0] for count in activation_count)
total_days_activated = len(activation_count)
good_trials_right = [count[0] for count in analysed_count]
good_trials_left = [count[1] for count in analysed_count]
total_good_trials_right = sum(good_trials_right)
total_good_trials_left = sum(good_trials_left)
print("Total number of exhibit activations: {total}".format(total=total_activation))
print("Total number of good right eye camera trials: {good_total}".format(good_total=total_good_trials_right))
print("Total number of good left eye camera trials: {good_total}".format(good_total=total_good_trials_left))
activation_array = np.array(activation_count)
analysed_array_right = np.array(good_trials_right)
analysed_array_left = np.array(good_trials_left)
### NEED BETTER PLOTS FOR EXHIBIT ENGAGEMENT
# activation based on: 
# day of the week
# time of the day
# month of the year
# language chosen
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
### PUPIL POSITION AND MOVEMENT ###
all_trials_position_right_data = [all_right_trials_contours_X, all_right_trials_contours_Y, all_right_trials_circles_X, all_right_trials_circles_Y]
all_trials_position_left_data = [all_left_trials_contours_X, all_left_trials_contours_Y, all_left_trials_circles_X, all_left_trials_circles_Y]
all_positions = [all_trials_position_right_data, all_trials_position_left_data]
# measure movement: setup
all_right_contours_movement_X = {key:[] for key in stim_vids}
all_right_circles_movement_X = {key:[] for key in stim_vids}
all_right_contours_movement_Y = {key:[] for key in stim_vids}
all_right_circles_movement_Y = {key:[] for key in stim_vids}
all_left_contours_movement_X = {key:[] for key in stim_vids}
all_left_circles_movement_X = {key:[] for key in stim_vids}
all_left_contours_movement_Y = {key:[] for key in stim_vids}
all_left_circles_movement_Y = {key:[] for key in stim_vids}
all_movement_right = [all_right_contours_movement_X, all_right_contours_movement_Y, all_right_circles_movement_X, all_right_circles_movement_Y]
all_movement_left = [all_left_contours_movement_X, all_left_contours_movement_Y, all_left_circles_movement_X, all_left_circles_movement_Y]
all_movements = [all_movement_right, all_movement_left]
side_names = ['Right', 'Left']
cAxis_names = ['contoursX', 'contoursY', 'circlesX', 'circlesY']
### CALCULATE MOVEMENT ###
for side in range(len(all_positions)):
    for c_axis in range(len(all_positions[side])):
        for stimuli in all_positions[side][c_axis]:
            print('Calculating movements for {side} side, {cAxis_type}, stimulus {stim}'.format(side=side_names[side], cAxis_type=cAxis_names[c_axis], stim=stimuli))
            # if there are nans (dropped frames) for more than 2 seconds of video time, then toss that trial
            dropped_frames_threshold = 2000/downsampled_bucket_size_ms
            all_movements[side][c_axis][stimuli] = calc_mvmnt_from_pos(all_positions[side][c_axis][stimuli], dropped_frames_threshold, 100, -100)
# measure motion: setup
all_right_contours_X_avg_motion = {key:[] for key in stim_vids}
all_right_circles_X_avg_motion = {key:[] for key in stim_vids}
all_right_contours_Y_avg_motion = {key:[] for key in stim_vids}
all_right_circles_Y_avg_motion = {key:[] for key in stim_vids}
all_left_contours_X_avg_motion = {key:[] for key in stim_vids}
all_left_circles_X_avg_motion = {key:[] for key in stim_vids}
all_left_contours_Y_avg_motion = {key:[] for key in stim_vids}
all_left_circles_Y_avg_motion = {key:[] for key in stim_vids}
all_avg_motion_right = [all_right_contours_X_avg_motion, all_right_contours_Y_avg_motion, all_right_circles_X_avg_motion, all_right_circles_Y_avg_motion]
all_avg_motion_left = [all_left_contours_X_avg_motion, all_left_contours_Y_avg_motion, all_left_circles_X_avg_motion, all_left_circles_Y_avg_motion]
all_avg_motion = [all_avg_motion_right, all_avg_motion_left]
# mark motion peaks: setup
all_RcontoursX_avg_motion_peaks = {key:[] for key in stim_vids}
all_RcirclesX_avg_motion_peaks = {key:[] for key in stim_vids}
all_RcontoursY_avg_motion_peaks = {key:[] for key in stim_vids}
all_RcirclesY_avg_motion_peaks = {key:[] for key in stim_vids}
all_LcontoursX_avg_motion_peaks = {key:[] for key in stim_vids}
all_LcirclesX_avg_motion_peaks = {key:[] for key in stim_vids}
all_LcontoursY_avg_motion_peaks = {key:[] for key in stim_vids}
all_LcirclesY_avg_motion_peaks = {key:[] for key in stim_vids}
all_avg_motion_right_peaks = [all_RcontoursX_avg_motion_peaks, all_RcontoursY_avg_motion_peaks, all_RcirclesX_avg_motion_peaks, all_RcirclesY_avg_motion_peaks]
all_avg_motion_left_peaks = [all_LcontoursX_avg_motion_peaks, all_LcontoursY_avg_motion_peaks, all_LcirclesX_avg_motion_peaks, all_LcirclesY_avg_motion_peaks]
all_avg_motion_peaks = [all_avg_motion_right_peaks, all_avg_motion_left_peaks]
### CALCULATE AVERAGE PIXEL MOTION PER TIME BUCKET FOR EACH STIM ###
for side in range(len(all_movements)):
    for c_axis in range(len(all_movements[side])):
        for stimuli in all_movements[side][c_axis]:
            print('Calculating average motion for {side} side, {cAxis_type}, stimulus {stim}'.format(side=side_names[side], cAxis_type=cAxis_names[c_axis], stim=stimuli))
            avg_motion_this_stim, peaks_this_stim = calc_avg_motion_and_peaks(all_movements[side][c_axis][stimuli], smoothing_window)
            all_avg_motion[side][c_axis][stimuli] = avg_motion_this_stim
            all_avg_motion_peaks[side][c_axis][stimuli] = peaks_this_stim
### mark saccades: setup
all_right_contours_X_saccades = {key:{} for key in stim_vids}
all_right_circles_X_saccades = {key:{} for key in stim_vids}
all_right_contours_Y_saccades = {key:{} for key in stim_vids}
all_right_circles_Y_saccades = {key:{} for key in stim_vids}
all_left_contours_X_saccades = {key:{} for key in stim_vids}
all_left_circles_X_saccades = {key:{} for key in stim_vids}
all_left_contours_Y_saccades = {key:{} for key in stim_vids}
all_left_circles_Y_saccades = {key:{} for key in stim_vids}
all_saccades_right = [all_right_contours_X_saccades, all_right_contours_Y_saccades, all_right_circles_X_saccades, all_right_circles_Y_saccades]
all_saccades_left = [all_left_contours_X_saccades, all_left_contours_Y_saccades, all_left_circles_X_saccades, all_left_circles_Y_saccades]
all_saccades = [all_saccades_right, all_saccades_left]
### FILTER INDIVIDUAL EYE MOVEMENTS FOR SACCADES ###
saccade_thresholds = [2.5, 5, 10, 20, 30, 40, 50, 60] # pixels
for side in range(len(all_movements)):
    for c_axis in range(len(all_movements[side])):
        for stim in all_movements[side][c_axis]:
            all_saccades[side][c_axis][stim] = {key:{} for key in saccade_thresholds}
            this_stim_N = len(all_movements[side][c_axis][stim])
            count_threshold = this_stim_N/10
            windowed_count_thresholds = [this_stim_N/(i*2) for i in range(1, len(saccade_thresholds)+1)]
            for thresh in range(len(saccade_thresholds)):
                print('Looking for movements greater than {p} pixels in {side} side, {cAxis_type}, stimulus {s}'.format(p=saccade_thresholds[thresh], side=side_names[side], cAxis_type=cAxis_names[c_axis], s=stim))
                saccades_window = 40 # timebuckets
                s_thresh = saccade_thresholds[thresh]
                w_thresh = windowed_count_thresholds[thresh]
                all_saccades[side][c_axis][stim][s_thresh] = find_saccades(all_movements[side][c_axis][stim], s_thresh, count_threshold, saccades_window, w_thresh)
### PUPIL SIZE ###
# average pupil diameters: setup
all_right_sizes = [all_right_trials_contours, all_right_trials_circles]
all_left_sizes = [all_left_trials_contours, all_left_trials_circles]
all_sizes = [all_right_sizes, all_left_sizes]
all_right_size_contours_means = {key:[] for key in stim_vids}
all_left_size_contours_means = {key:[] for key in stim_vids}
all_right_size_circles_means = {key:[] for key in stim_vids}
all_left_size_circles_means = {key:[] for key in stim_vids}
all_right_size_means = [all_right_size_contours_means, all_right_size_circles_means]
all_left_size_means = [all_left_size_contours_means, all_left_size_circles_means]
all_size_means = [all_right_size_means, all_left_size_means]
# find peaks in pupil diameter sizes: setup
all_right_size_contours_peaks = {key:[] for key in stim_vids}
all_left_size_contours_peaks = {key:[] for key in stim_vids}
all_right_size_circles_peaks = {key:[] for key in stim_vids}
all_left_size_circles_peaks = {key:[] for key in stim_vids}
all_right_size_peaks = [all_right_size_contours_peaks, all_right_size_circles_peaks]
all_left_size_peaks = [all_left_size_contours_peaks, all_left_size_circles_peaks]
all_size_peaks = [all_right_size_peaks, all_left_size_peaks]
# Compute global mean
for side in range(len(all_sizes)):
    for i in range(len(all_sizes[side])):
        for stimulus in all_sizes[side][i].keys(): 
            print('Calculating average pupil sizes for right camera, {c}, stimulus {s}'.format(c=cType_names[i],s=stimulus))
            avg_pupil_size = np.nanmean(all_sizes[side][i][stimulus], 0)
            avg_pupil_size_smoothed = savgol_filter(avg_pupil_size, smoothing_window, 3)
            all_size_means[side][i][stimulus] = avg_pupil_size_smoothed
            avg_pupil_size_peaks, _ = find_peaks(avg_pupil_size_smoothed, prominence=0.125)
            all_size_peaks[side][i][stimulus] = avg_pupil_size_peaks
### ------------------------------ ###
### POOL ACROSS CALBIRATION SEQUENCE
# all pupil movements
all_right_contours_movement_X_cal = []
all_right_circles_movement_X_cal = []
all_right_contours_movement_Y_cal = []
all_right_circles_movement_Y_cal = []
all_left_contours_movement_X_cal = []
all_left_circles_movement_X_cal = []
all_left_contours_movement_Y_cal = []
all_left_circles_movement_Y_cal = []
all_movement_right_cal = [all_right_contours_movement_X_cal, all_right_contours_movement_Y_cal, all_right_circles_movement_X_cal, all_right_circles_movement_Y_cal]
all_movement_left_cal = [all_left_contours_movement_X_cal, all_left_contours_movement_Y_cal, all_left_circles_movement_X_cal, all_left_circles_movement_Y_cal]
all_movements_cal = [all_movement_right_cal, all_movement_left_cal]
for side in range(len(all_movements_cal)):
    for i in range(len(all_movements_cal[side])):
        pool_pupil_data_for_global_moment_of_interest(all_movements[side][i], all_cal_avg_tbuckets, all_movements_cal[side][i])
# pupil avg motion
all_right_contours_X_avg_motion_cal = []
all_right_circles_X_avg_motion_cal = []
all_right_contours_Y_avg_motion_cal = []
all_right_circles_Y_avg_motion_cal = []
all_left_contours_X_avg_motion_cal = []
all_left_circles_X_avg_motion_cal = []
all_left_contours_Y_avg_motion_cal = []
all_left_circles_Y_avg_motion_cal = []
all_avg_motion_right_cal = [all_right_contours_X_avg_motion_cal, all_right_contours_Y_avg_motion_cal, all_right_circles_X_avg_motion_cal, all_right_circles_Y_avg_motion_cal]
all_avg_motion_left_cal = [all_left_contours_X_avg_motion_cal, all_left_contours_Y_avg_motion_cal, all_left_circles_X_avg_motion_cal, all_left_circles_Y_avg_motion_cal]
all_avg_motion_cal = [all_avg_motion_right_cal, all_avg_motion_left_cal]
for side in range(len(all_avg_motion_cal)):
    for i in range(len(all_avg_motion_cal[side])):
        pool_pupil_means_for_global_moment_of_interest(all_avg_motion[side][i], all_cal_avg_tbuckets, all_avg_motion_cal[side][i])
# pupil avg motion peaks - NOT SORTED, WITH DUPLICATES
all_RcontoursX_avg_motion_peaks_cal = []
all_RcirclesX_avg_motion_peaks_cal = []
all_RcontoursY_avg_motion_peaks_cal = []
all_RcirclesY_avg_motion_peaks_cal = []
all_LcontoursX_avg_motion_peaks_cal = []
all_LcirclesX_avg_motion_peaks_cal = []
all_LcontoursY_avg_motion_peaks_cal = []
all_LcirclesY_avg_motion_peaks_cal = []
all_avg_motion_right_peaks_cal = [all_RcontoursX_avg_motion_peaks_cal, all_RcontoursY_avg_motion_peaks_cal, all_RcirclesX_avg_motion_peaks_cal, all_RcirclesY_avg_motion_peaks_cal]
all_avg_motion_left_peaks_cal = [all_LcontoursX_avg_motion_peaks_cal, all_LcontoursY_avg_motion_peaks_cal, all_LcirclesX_avg_motion_peaks_cal, all_LcirclesY_avg_motion_peaks_cal]
all_avg_motion_peaks_cal = [all_avg_motion_right_peaks_cal, all_avg_motion_left_peaks_cal]
for side in range(len(all_avg_motion_peaks_cal)):
    for i in range(len(all_avg_motion_peaks_cal[side])):
        pool_pupil_peaks_for_global_moment_of_interest(all_avg_motion_peaks[side][i], all_cal_avg_tbuckets, all_avg_motion_peaks_cal[side][i])
# pupil saccades
all_right_contours_X_saccades_cal = {thresh:{} for thresh in saccade_thresholds}
all_right_circles_X_saccades_cal = {thresh:{} for thresh in saccade_thresholds}
all_right_contours_Y_saccades_cal = {thresh:{} for thresh in saccade_thresholds}
all_right_circles_Y_saccades_cal = {thresh:{} for thresh in saccade_thresholds}
all_left_contours_X_saccades_cal = {thresh:{} for thresh in saccade_thresholds}
all_left_circles_X_saccades_cal = {thresh:{} for thresh in saccade_thresholds}
all_left_contours_Y_saccades_cal = {thresh:{} for thresh in saccade_thresholds}
all_left_circles_Y_saccades_cal = {thresh:{} for thresh in saccade_thresholds}
all_saccades_right_cal = [all_right_contours_X_saccades_cal, all_right_contours_Y_saccades_cal, all_right_circles_X_saccades_cal, all_right_circles_Y_saccades_cal]
all_saccades_left_cal = [all_left_contours_X_saccades_cal, all_left_contours_Y_saccades_cal, all_left_circles_X_saccades_cal, all_left_circles_Y_saccades_cal]
all_saccades_cal = [all_saccades_right_cal, all_saccades_left_cal]
for side in range(len(all_saccades_cal)):
    for i in range(len(all_saccades_cal[side])):
        pool_pupil_saccades_for_global_moment_of_interest(all_saccades[side][i], all_cal_avg_tbuckets, all_saccades_cal[side][i])
# all pupil sizes
all_right_contours_calibration = []
all_right_circles_calibration = []
all_left_contours_calibration = []
all_left_circles_calibration = []
all_right_sizes_calibration = [all_right_contours_calibration, all_right_circles_calibration]
all_left_sizes_calibration = [all_left_contours_calibration, all_left_circles_calibration]
all_sizes_calibration = [all_right_sizes_calibration, all_left_sizes_calibration]
for side in range(len(all_sizes_calibration)):
    for i in range(len(all_sizes_calibration[side])):
        pool_pupil_data_for_global_moment_of_interest(all_sizes[side][i], all_cal_avg_tbuckets, all_sizes_calibration[side][i])
# pupil size means
all_right_contours_means_calibration = []
all_right_circles_means_calibration = []
all_left_contours_means_calibration = []
all_left_circles_means_calibration = []
all_right_sizes_means_cal = [all_right_contours_means_calibration, all_right_circles_means_calibration]
all_left_sizes_means_cal = [all_left_contours_means_calibration, all_left_circles_means_calibration]
all_sizes_means_cal = [all_right_sizes_means_cal, all_left_sizes_means_cal]
for side in range(len(all_sizes_means_cal)):
    for i in range(len(all_sizes_means_cal[side])):
        pool_pupil_means_for_global_moment_of_interest(all_size_means[side][i], all_cal_avg_tbuckets, all_sizes_means_cal[side][i])
# pupil size mean peaks
all_right_contours_peaks_calibration = []
all_right_circles_peaks_calibration = []
all_left_contours_peaks_calibration = []
all_left_circles_peaks_calibration = []
all_right_peaks_cal = [all_right_contours_peaks_calibration, all_right_circles_peaks_calibration]
all_left_peaks_cal = [all_left_contours_peaks_calibration, all_left_circles_peaks_calibration]
all_sizes_peaks_cal = [all_right_peaks_cal, all_left_peaks_cal]
for side in range(len(all_sizes_peaks_cal)):
    for i in range(len(all_sizes_peaks_cal[side])):
        pool_pupil_peaks_for_global_moment_of_interest(all_size_peaks[side][i], all_cal_avg_tbuckets, all_sizes_peaks_cal[side][i])


# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
### GLOBAL VARIABLES FOR PLOTTING DATA ###
fig_size = 200 # dpi
image_type_options = ['.png', '.pdf']
plotting_saccades_window = 40 # MAKE SURE THIS ==saccades_window!!
cType_names = ['Contours', 'Circles']
plot_lum_types = {'calibration':all_cal_avg_lum_smoothed_baselined,'octopus':all_octo_avg_lum_smoothed_baselined,'unique':all_stims_unique_avg_lum_smoothed_baselined}
### ------------------------------ ###
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
#### EVERYTHING BELOW HERE IS UNDER CONSTRUCTION!!!! ###
### GENERATE PLOTS ###
# Plot pupil sizes
# calibration
pupil_analysis_type_name = cType_names[0]
plot_type_right = all_right_trials_contours
plot_means_right = all_right_size_contours_means
plot_means_right_peaks = all_right_size_contours_peaks
plot_type_left = all_left_trials_contours
plot_means_left = all_left_size_contours_means
plot_means_left_peaks = all_left_size_contours_peaks
plot_luminance = all_cal_avg_lum_smoothed_baselined
# fig name and path
figure_name = 'PupilSizes_' + 'calibration' + '_' + pupil_analysis_type_name + '_' + todays_datetime + '_dpi' + str(fig_size) + '.png' 
figure_path = os.path.join(pupils_folder, figure_name)
figure_title = "Pupil sizes of participants during calibration sequence \n" + str(total_activation) + " total exhibit activations" + "\nAnalysis type: " + pupil_analysis_type_name + "\nPlotted on " + todays_datetime
# draw fig
plt.figure(figsize=(14, 14), dpi=fig_size)
plt.suptitle(figure_title, fontsize=12, y=0.98)
# subplot: Right eye sizes
plt.subplot(3,1,1)
plt.ylabel('Percent change in pupil area (from baseline)', fontsize=11)
plt.title('Right eye pupil sizes; N = ' + str(plot_N_right), fontsize=10, color='grey', style='italic')
plt.minorticks_on()
plt.grid(b=True, which='major', linestyle='--')
plt.plot(plot_type_right.T, '.', MarkerSize=1, color=[1.0, 0.98, 0.0, 0.005])
plt.plot(plot_means_right, linewidth=1.5, color=[0.9686, 0.0, 1.0, 0.75])
for peak in plot_means_right_peaks:
    if peak<1250:
        plt.plot(peak, plot_means_right[peak], 'x')
        plt.text(peak-15, plot_means_right[peak]+0.25, str(peak), fontsize='xx-small', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
plt.xlim(-10,1250)
plt.ylim(-1,1)
# subplot: Left eye sizes
plt.subplot(3,1,2)
plt.ylabel('Percent change in pupil area (from baseline)', fontsize=11)
plt.title('Left eye pupil sizes; N = ' + str(plot_N_left), fontsize=10, color='grey', style='italic')
plt.minorticks_on()
plt.grid(b=True, which='major', linestyle='--')
plt.plot(plot_type_left.T, '.', MarkerSize=1, color=[0.012, 0.7, 1.0, 0.005])
plt.plot(plot_means_left, linewidth=1.5, color=[1.0, 0.34, 0.012, 0.75])
for peak in plot_means_left_peaks:
    if peak<1250:
        plt.plot(peak, plot_means_left[peak], 'x')
        plt.text(peak-15, plot_means_left[peak]+0.25, str(peak), fontsize='xx-small', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
plt.xlim(-10,1250)
plt.ylim(-1,1)
# subplot: Average luminance of stimuli video
plt.subplot(3,1,3)
plt.ylabel('Percent change in luminance (from baseline)', fontsize=11)
plt.xlabel('Time buckets (downsampled, 1 time bucket = ' + str(downsampled_bucket_size_ms) + 'ms)', fontsize=11)
plt.title('Average luminance of ' + stim_name + ' as seen by world camera, grayscaled; N = ' + str(len(luminances[stim_type])), fontsize=10, color='grey', style='italic')
plt.grid(b=True, which='major', linestyle='--')
plt.plot(plot_luminance, linewidth=1, color=[0.192, 0.75, 0.004, 1])
for peak in plot_luminance_peaks:
    plt.plot(peak, plot_luminance[peak], 'x')
    plt.text(peak-15, plot_luminance[peak]+0.5, str(peak), fontsize='xx-small', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
plt.xlim(-10,1250)
plt.ylim(-1,7)
# save and display
plt.subplots_adjust(hspace=0.5)
plt.savefig(figure_path)
plt.show(block=False)
plt.pause(1)
plt.close()

#### EVERYTHING BELOW HERE IS UNDER CONSTRUCTION!!!! ###
for stim_type in stim_vids: 
    for i in range(len(all_right_sizes)): 
        plot_type_right = np.array(all_right_sizes[i][stim_type])
        plot_N_right = len(all_right_sizes[i][stim_type])
        plot_type_left = np.array(all_left_sizes[i][stim_type])
        plot_N_left = len(all_left_sizes[i][stim_type])
        plot_means_right = all_right_size_means[i][stim_type]
        plot_means_right_peaks = all_right_size_peaks[i][stim_type]
        plot_means_left = all_left_size_means[i][stim_type]
        plot_means_left_peaks = all_left_size_peaks[i][stim_type]
        plot_luminance = luminances_avg[stim_type]
        plot_luminance_peaks = luminances_peaks[stimuli]
        plot_type_name = plot_types[i]
        stim_name = stim_float_to_name[stim_type]
        # fig name and path
        figure_name = 'AveragePupilSizes_' + plot_type_name + '_' + stim_name + '_' + todays_datetime + '_dpi' + str(fig_size) + '.png' 
        figure_path = os.path.join(pupils_folder, figure_name)
        figure_title = "Pupil sizes of participants \n" + str(total_activation) + " total exhibit activations" + "\nAnalysis type: " + plot_type_name + "\nStimulus type: " + stim_name + "\nPlotted on " + todays_datetime
        # draw fig
        plt.figure(figsize=(14, 14), dpi=fig_size)
        plt.suptitle(figure_title, fontsize=12, y=0.98)
        # subplot: Right eye sizes
        plt.subplot(3,1,1)
        plt.ylabel('Percent change in pupil area (from baseline)', fontsize=11)
        plt.title('Right eye pupil sizes; N = ' + str(plot_N_right), fontsize=10, color='grey', style='italic')
        plt.minorticks_on()
        plt.grid(b=True, which='major', linestyle='--')
        plt.plot(plot_type_right.T, '.', MarkerSize=1, color=[1.0, 0.98, 0.0, 0.005])
        plt.plot(plot_means_right, linewidth=1.5, color=[0.9686, 0.0, 1.0, 0.75])
        for peak in plot_means_right_peaks:
            if peak<1250:
                plt.plot(peak, plot_means_right[peak], 'x')
                plt.text(peak-15, plot_means_right[peak]+0.25, str(peak), fontsize='xx-small', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        plt.xlim(-10,1250)
        plt.ylim(-1,1)
        # subplot: Left eye sizes
        plt.subplot(3,1,2)
        plt.ylabel('Percent change in pupil area (from baseline)', fontsize=11)
        plt.title('Left eye pupil sizes; N = ' + str(plot_N_left), fontsize=10, color='grey', style='italic')
        plt.minorticks_on()
        plt.grid(b=True, which='major', linestyle='--')
        plt.plot(plot_type_left.T, '.', MarkerSize=1, color=[0.012, 0.7, 1.0, 0.005])
        plt.plot(plot_means_left, linewidth=1.5, color=[1.0, 0.34, 0.012, 0.75])
        for peak in plot_means_left_peaks:
            if peak<1250:
                plt.plot(peak, plot_means_left[peak], 'x')
                plt.text(peak-15, plot_means_left[peak]+0.25, str(peak), fontsize='xx-small', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        plt.xlim(-10,1250)
        plt.ylim(-1,1)
        # subplot: Average luminance of stimuli video
        plt.subplot(3,1,3)
        plt.ylabel('Percent change in luminance (from baseline)', fontsize=11)
        plt.xlabel('Time buckets (downsampled, 1 time bucket = ' + str(downsampled_bucket_size_ms) + 'ms)', fontsize=11)
        plt.title('Average luminance of ' + stim_name + ' as seen by world camera, grayscaled; N = ' + str(len(luminances[stim_type])), fontsize=10, color='grey', style='italic')
        plt.grid(b=True, which='major', linestyle='--')
        plt.plot(plot_luminance, linewidth=1, color=[0.192, 0.75, 0.004, 1])
        for peak in plot_luminance_peaks:
            plt.plot(peak, plot_luminance[peak], 'x')
            plt.text(peak-15, plot_luminance[peak]+0.5, str(peak), fontsize='xx-small', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        plt.xlim(-10,1250)
        plt.ylim(-1,7)
        # save and display
        plt.subplots_adjust(hspace=0.5)
        plt.savefig(figure_path)
        plt.show(block=False)
        plt.pause(1)
        plt.close()
# plot movement traces
all_movement_right_plot = [(all_right_contours_movement_X, all_right_contours_movement_Y), (all_right_circles_movement_X, all_right_circles_movement_Y)]
all_movement_left_plot = [(all_left_contours_movement_X, all_left_contours_movement_Y), (all_left_circles_movement_X, all_left_circles_movement_Y)]
all_movements_plot = [all_movement_right_plot, all_movement_left_plot]
for side in range(len(all_movements_plot)):
    for c_type in range(len(all_movements_plot[side])):
        for stimuli in all_movements_plot[side][c_type][0]:
            plot_type_name = side_names[side] + cType_names[c_type]
            stim_name = stim_float_to_name[stimuli]
            plot_type_X = all_movements_plot[side][c_type][0][stimuli]
            plot_N_X = len(plot_type_X)
            plot_type_Y = all_movements_plot[side][c_type][1][stimuli]
            plot_N_Y = len(plot_type_Y)
            plot_luminance = luminances_avg[stimuli]
            plot_luminance_peaks = luminances_peaks[stimuli]
            # fig name and path
            figure_name = 'MovementTraces_' + plot_type_name + '_' + stim_name + '_' + todays_datetime + '_dpi' + str(fig_size) + '.png' 
            figure_path = os.path.join(pupils_folder, figure_name)
            figure_title = "Pupil movement of participants \n" + str(total_activation) + " total exhibit activations" + "\nAnalysis type: " + plot_type_name + "\nStimulus type: " + stim_name + "\nPlotted on " + todays_datetime
            # draw fig
            plt.figure(figsize=(14, 14), dpi=fig_size)
            plt.suptitle(figure_title, fontsize=12, y=0.98)
            # x-axis
            plt.subplot(3,1,1)
            plt.ylabel('Change in pixels', fontsize=11)
            plt.title('Pupil movement in the X-axis; N = ' + str(plot_N_X), fontsize=10, color='grey', style='italic')
            plt.minorticks_on()
            plt.grid(b=True, which='major', linestyle='--')
            for trial in plot_type_X:
                plt.plot(trial, linewidth=0.5, color=[0.86, 0.27, 1.0, 0.005])
            plt.xlim(-10,1250)
            plt.ylim(-80,80)
            # y-axis
            plt.subplot(3,1,2)
            plt.ylabel('Change in pixels', fontsize=11)
            plt.title('Pupil movement in the Y-axis; N = ' + str(plot_N_Y), fontsize=10, color='grey', style='italic')
            plt.minorticks_on()
            plt.grid(b=True, which='major', linestyle='--')
            for trial in plot_type_Y:
                plt.plot(trial, linewidth=0.5, color=[0.25, 0.25, 1.0, 0.005])
            plt.xlim(-10,1250)
            plt.ylim(-80,80)
            # luminance
            plt.subplot(3,1,3)
            plt.ylabel('Percent change in luminance', fontsize=11)
            plt.xlabel('Time buckets (downsampled, 1 time bucket = ' + str(downsampled_bucket_size_ms) + 'ms)', fontsize=11)
            plt.title('Average luminance of ' + stim_name + ' as seen by world camera, grayscaled; N = ' + str(len(luminances[stimuli])), fontsize=10, color='grey', style='italic')
            plt.grid(b=True, which='major', linestyle='--')
            plt.plot(plot_luminance, linewidth=0.75, color=[1.0, 0.13, 0.4, 1])
            for peak in plot_luminance_peaks:
                plt.plot(peak, plot_luminance[peak], 'x')
                plt.text(peak-15, plot_luminance[peak]+0.5, str(peak), fontsize='xx-small', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
            plt.xlim(-10,1250)
            plt.ylim(-1,7)
            # save and display
            plt.subplots_adjust(hspace=0.5)
            plt.savefig(figure_path)
            plt.show(block=False)
            plt.pause(1)
            plt.close()
# plot MOTION traces (abs val of movement traces)
all_avg_motion_right_plot = [(all_right_contours_X_avg_motion, all_right_contours_Y_avg_motion), (all_right_circles_X_avg_motion, all_right_circles_Y_avg_motion)]
all_avg_motion_left_plot = [(all_left_contours_X_avg_motion, all_left_contours_Y_avg_motion), (all_left_circles_X_avg_motion, all_left_circles_Y_avg_motion)]
all_avg_motion_plot = [all_avg_motion_right_plot, all_avg_motion_left_plot]
all_avg_motion_right_peaks_plot = [(all_RcontoursX_avg_motion_peaks, all_RcontoursY_avg_motion_peaks), (all_RcirclesX_avg_motion_peaks, all_RcirclesY_avg_motion_peaks)]
all_avg_motion_left_peaks_plot = [(all_LcontoursX_avg_motion_peaks, all_LcontoursY_avg_motion_peaks), (all_LcirclesX_avg_motion_peaks, all_LcirclesY_avg_motion_peaks)]
all_avg_motion_peaks_plot = [all_avg_motion_right_peaks_plot, all_avg_motion_left_peaks_plot]
for side in range(len(all_movements_plot)):
    for c_type in range(len(all_movements_plot[side])):
        for stimuli in all_movements_plot[side][c_type][0]:
            plot_type_name = side_names[side] + cType_names[c_type]
            stim_name = stim_float_to_name[stimuli]
            plot_type_X = all_movements_plot[side][c_type][0][stimuli]
            plot_type_X_avg = all_avg_motion_plot[side][c_type][0][stimuli]
            plot_type_X_avg_peaks = all_avg_motion_peaks_plot[side][c_type][0][stimuli]
            plot_N_X = len(plot_type_X)
            plot_type_Y = all_movements_plot[side][c_type][1][stimuli]
            plot_type_Y_avg = all_avg_motion_plot[side][c_type][1][stimuli]
            plot_type_Y_avg_peaks = all_avg_motion_peaks_plot[side][c_type][1][stimuli]
            plot_N_Y = len(plot_type_Y)
            plot_luminance = luminances_avg[stimuli]
            plot_luminance_peaks = luminances_peaks[stimuli]
            # fig name and path
            figure_name = 'MotionTraces-AvgMotionPeaks' + str(plotting_saccades_window) + '_' + plot_type_name + '_' + stim_name + '_' + todays_datetime + '_dpi' + str(fig_size) + '.png' 
            figure_path = os.path.join(pupils_folder, figure_name)
            figure_title = "Pupil motion of participants \n" + str(total_activation) + " total exhibit activations" + "\nAnalysis type: " + plot_type_name + "\nStimulus type: " + stim_name + "\nPeak finding window: " + str(plotting_saccades_window) + "\nPlotted on " + todays_datetime
            # draw fig
            plt.figure(figsize=(14, 14), dpi=fig_size)
            plt.suptitle(figure_title, fontsize=12, y=0.98)
            # x-axis
            plt.subplot(3,1,1)
            plt.ylabel('Change in pixels', fontsize=11)
            plt.title('Pupil movement in the X-axis; N = ' + str(plot_N_X), fontsize=10, color='grey', style='italic')
            plt.minorticks_on()
            plt.grid(b=True, which='major', linestyle='--')
            for trial in plot_type_X:
                plt.plot(abs(trial), linewidth=0.5, color=[0.86, 0.27, 1.0, 0.005])
            plt.plot(plot_type_X_avg, linewidth=1, color=[0.4, 1.0, 0.27, 1])
            for peak in plot_type_X_avg_peaks:
                if peak<1250:
                    plt.plot(peak, plot_type_X_avg[peak], 'x')
                    plt.text(peak-15, plot_type_X_avg[peak]+5, str(peak), fontsize='xx-small', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
            plt.xlim(-10,1250)
            plt.ylim(-5,40)
            # y-axis
            plt.subplot(3,1,2)
            plt.ylabel('Change in pixels', fontsize=11)
            plt.title('Pupil movement in the Y-axis; N = ' + str(plot_N_Y), fontsize=10, color='grey', style='italic')
            plt.minorticks_on()
            plt.grid(b=True, which='major', linestyle='--')
            for trial in plot_type_Y:
                plt.plot(abs(trial), linewidth=0.5, color=[0.25, 0.25, 1.0, 0.005])
            plt.plot(plot_type_Y_avg, linewidth=1, color=[1.0, 1.0, 0.25, 1])
            for peak in plot_type_Y_avg_peaks:
                if peak<1250:
                    plt.plot(peak, plot_type_Y_avg[peak], 'x')
                    plt.text(peak-15, plot_type_Y_avg[peak]+5, str(peak), fontsize='xx-small', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
            plt.xlim(-10,1250)
            plt.ylim(-5,40)
            # luminance
            plt.subplot(3,1,3)
            plt.ylabel('Percent change in luminance', fontsize=11)
            plt.xlabel('Time buckets (downsampled, 1 time bucket = ' + str(downsampled_bucket_size_ms) + 'ms)', fontsize=11)
            plt.title('Average luminance of ' + stim_name + ' as seen by world camera, grayscaled; N = ' + str(len(luminances[stimuli])), fontsize=10, color='grey', style='italic')
            plt.grid(b=True, which='major', linestyle='--')
            plt.plot(plot_luminance, linewidth=1, color=[1.0, 0.13, 0.4, 1])
            for peak in plot_luminance_peaks:
                plt.plot(peak, plot_luminance[peak], 'x')
                plt.text(peak-15, plot_luminance[peak]+0.5, str(peak), fontsize='xx-small', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
            plt.xlim(-10,1250)
            plt.ylim(-1,7)
            # save and display
            plt.subplots_adjust(hspace=0.5)
            plt.savefig(figure_path)
            plt.show(block=False)
            plt.pause(1)
            plt.close()
# plot peaks/saccades
all_peaks_right_plot = [(all_right_contours_X_peaks, all_right_contours_Y_peaks), (all_right_circles_X_peaks, all_right_circles_Y_peaks)]
all_peaks_left_plot = [(all_left_contours_X_peaks, all_left_contours_Y_peaks), (all_left_circles_X_peaks, all_left_circles_Y_peaks)]
all_peaks_plot = [all_peaks_right_plot, all_peaks_left_plot]
for side in range(len(all_movements_plot)):
    for c_type in range(len(all_movements_plot[side])):
        for stimuli in all_movements_plot[side][c_type][0]:
            plot_type_name = side_names[side] + cType_names[c_type]
            stim_name = stim_float_to_name[stimuli]
            plot_type_X = all_movements_plot[side][c_type][0][stimuli]
            plot_type_X_peaks = all_peaks_plot[side][c_type][0][stimuli]
            plot_N_X = len(plot_type_X)
            plot_type_Y = all_movements_plot[side][c_type][1][stimuli]
            plot_type_Y_peaks = all_peaks_plot[side][c_type][1][stimuli]
            plot_N_Y = len(plot_type_Y)
            plot_luminance = luminances_avg[stimuli]
            plot_luminance_peaks = luminances_peaks[stimuli]
            # fig name and path
            figure_name = 'MotionTraces-Saccades' + str(plotting_saccades_window) + '_' + plot_type_name + '_' + stim_name + '_' + todays_datetime + '_dpi' + str(fig_size) + '.png' 
            figure_path = os.path.join(pupils_folder, figure_name)
            figure_title = "Pupil motion of participants \n" + str(total_activation) + " total exhibit activations" + "\nAnalysis type: " + plot_type_name + "\nStimulus type: " + stim_name + "\nPeaks plotted at height of pixel movement threshold, peak finding window: " + str(plotting_saccades_window) + "\nPlotted on " + todays_datetime
            # begin drawing fig
            plt.figure(figsize=(14, 14), dpi=fig_size)
            plt.suptitle(figure_title, fontsize=12, y=0.98)
            # x-axis
            plt.subplot(3,1,1)
            plt.ylabel('Change in pixels', fontsize=11)
            plt.title('Pupil movement in the X-axis; N = ' + str(plot_N_X), fontsize=10, color='grey', style='italic')
            plt.minorticks_on()
            plt.grid(b=True, which='major', linestyle='--')
            for trial in plot_type_X:
                plt.plot(abs(trial), linewidth=0.5, color=[0.86, 0.27, 1.0, 0.005])
            for threshold in plot_type_X_peaks.keys():
                for key in plot_type_X_peaks[threshold].keys():
                    if key<1250:
                        plt.plot(key, threshold, '1', color=[0.4, 1.0, 0.27, 1.0])
            plt.xlim(-10,1250)
            plt.ylim(-5,60)
            # y-axis
            plt.subplot(3,1,2)
            plt.ylabel('Change in pixels', fontsize=11)
            plt.title('Pupil movement in the Y-axis; N = ' + str(plot_N_Y), fontsize=10, color='grey', style='italic')
            plt.minorticks_on()
            plt.grid(b=True, which='major', linestyle='--')
            for trial in plot_type_Y:
                plt.plot(abs(trial), linewidth=0.5, color=[0.25, 0.25, 1.0, 0.005])
            for threshold in plot_type_Y_peaks.keys():
                for key in plot_type_Y_peaks[threshold].keys():
                    if key<1250:
                        plt.plot(key, threshold, '1', color=[1.0, 1.0, 0.25, 1.0])
            plt.xlim(-10,1250)
            plt.ylim(-5,60)
            # luminance
            plt.subplot(3,1,3)
            plt.ylabel('Percent change in luminance (from baseline)', fontsize=11)
            plt.xlabel('Time buckets (downsampled, 1 time bucket = ' + str(downsampled_bucket_size_ms) + 'ms)', fontsize=11)
            plt.title('Average luminance of ' + stim_name + ' as seen by world camera, grayscaled; N = ' + str(len(luminances[stimuli])), fontsize=10, color='grey', style='italic')
            plt.grid(b=True, which='major', linestyle='--')
            plt.plot(plot_luminance, linewidth=1, color=[1.0, 0.13, 0.4, 1])
            for peak in plot_luminance_peaks:
                plt.plot(peak, plot_luminance[peak], 'x')
                plt.text(peak-15, plot_luminance[peak]+0.5, str(peak), fontsize='xx-small', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
            plt.xlim(-10,1250)
            plt.ylim(-1,7)
            # save and display
            plt.subplots_adjust(hspace=0.5)
            plt.savefig(figure_path)
            plt.show(block=False)
            plt.pause(1)
            plt.close()
#FIN