### --------------------------------------------------------------------------- ###
# loads world vid luminance csv files 
# outputs normalized pupil size as binary files
# creates a scatter plot comparing luminance to pupil size
### --------------------------------------------------------------------------- ###
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
from collections import defaultdict
from scipy import stats
# set up log file to store all printed messages
#current_working_directory = os.getcwd()

###################################
# FUNCTIONS
###################################
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
        # iterate through trials
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

def phaseMeans_withDelay(delay_tb, normedPupils_array, calibLen_tb, allUniqueLens_tb, octoLen_tb):
    allCalib = []
    allOcto = []
    allUnique = []
    # Split trials into calib, octo, and unique
    for i, uniqueStim in enumerate(normedPupils_array):
        thisUnique = []
        uniqueLen_tb = allUniqueLens_tb[i]
        for normed_trial in uniqueStim:
            thisTrial_calib = normed_trial[delay_tb : delay_tb+calibLen_tb]
            allCalib.append(thisTrial_calib)
            thisTrial_unique = normed_trial[delay_tb+calibLen_tb+1 : delay_tb+calibLen_tb+1+uniqueLen_tb]
            thisUnique.append(thisTrial_unique)
            thisTrial_octo = normed_trial[delay_tb+calibLen_tb+1+uniqueLen_tb+1 : delay_tb+calibLen_tb+1+uniqueLen_tb+1+octoLen_tb]
            allOcto.append(thisTrial_octo)
        allUnique.append(thisUnique)
    calib_mean = np.nanmean(allCalib, axis=0)
    octo_mean = np.nanmean(allOcto, axis=0)
    unique_means = []
    for unique in allUnique:
        thisUnique_mean = np.nanmean(unique, axis=0)
        unique_means.append(thisUnique_mean)
    return calib_mean, octo_mean, unique_means

def leastSquares_pupilSize_lum(pupilSize_array, lum_array):
    # remove tb where pupil sizes are nans
    meanPupil_nonan = pupilSize_array[np.logical_not(np.isnan(pupilSize_array))]
    meanLum_nonan = lum_array[np.logical_not(np.isnan(pupilSize_array))]
    # remove tb where luminances are nans
    meanPupil_nonan = meanPupil_nonan[np.logical_not(np.isnan(meanLum_nonan))]
    meanLum_nonan = meanLum_nonan[np.logical_not(np.isnan(meanLum_nonan))]
    # calculate least squares regression line
    slope, intercept, rval, pval, stderr = stats.linregress(meanLum_nonan, meanPupil_nonan)
    return slope, intercept, rval, pval, stderr

def LumVsPupilSize_ScatterLinRegress(lum_array, pupilSize_array, phase_name, eyeAnalysis_name, pupilDelay_ms, save_folder):
    # make sure pupil size and world cam lum arrays are same size
    plotting_numTB = min(len(lum_array), len(pupilSize_array))
    lum_plot = lum_array[:plotting_numTB]
    pupil_plot = pupilSize_array[:plotting_numTB]
    # calculate least squares regression line
    slope, intercept, rval, pval, stderr = leastSquares_pupilSize_lum(pupilSize_array, lum_array)
    # figure path and title
    figPath = os.path.join(save_folder, '%s_meanLum-mean%s_delay%dms.png'%(phase_name, eyeAnalysis_name, pupilDelay_ms))
    figTitle = 'Mean luminance of world cam vs mean pupil size (%s) during %s, pupil delay = %dms'%(eyeAnalysis_name, phase_name, pupilDelay_ms)
    print('Plotting %s'%(figTitle))
    # draw scatter plot
    plt.figure(figsize=(9, 9), dpi=200)
    plt.suptitle(figTitle, fontsize=12, y=0.98)
    plt.ylabel('Mean pupil size (percent change from median of full trial)')
    plt.xlabel('Mean luminance of world cam')
    plt.plot(lum_plot, pupil_plot, '.', label='original data')
    # draw regression line
    plt.plot(lum_plot, intercept+slope*lum_plot, 'r', label='fitted line, r-squared: %f'%(rval**2))
    plt.legend()
    plt.savefig(figPath)
    plt.close()
    return rval

def splitPupils_withDelay_plotScatterLinRegress(delay_tb, downsample_ms, lum_array, pupilSize_array, calibLen_tb, uniqueLens_tb, octoLen_tb, eyeAnalysis_name, savePlotsFolder, saveDataFolder):
    # split normalized pupil size data into trial phases
    pupil_calib_mean, pupil_octo_mean, pupil_unique_means = phaseMeans_withDelay(delay_tb, pupilSize_array, calibLen_tb, uniqueLens_tb, octoLen_tb)
    # save normalized, split and averaged pupil size data as intermediate files
    ## output path
    calib_output = saveDataFolder + os.sep + 'meanNormedPupilSize_calib_%dmsDelay_%s.npy'%(delay_tb*downsample_ms,eyeAnalysis_name)
    octo_output = saveDataFolder + os.sep + 'meanNormedPupilSize_octo_%dmsDelay_%s.npy'%(delay_tb*downsample_ms,eyeAnalysis_name)
    unique1_output = saveDataFolder + os.sep + 'meanNormedPupilSize_u1_%dmsDelay_%s.npy'%(delay_tb*downsample_ms,eyeAnalysis_name)
    unique2_output = saveDataFolder + os.sep + 'meanNormedPupilSize_u2_%dmsDelay_%s.npy'%(delay_tb*downsample_ms,eyeAnalysis_name)
    unique3_output = saveDataFolder + os.sep + 'meanNormedPupilSize_u3_%dmsDelay_%s.npy'%(delay_tb*downsample_ms,eyeAnalysis_name)
    unique4_output = saveDataFolder + os.sep + 'meanNormedPupilSize_u4_%dmsDelay_%s.npy'%(delay_tb*downsample_ms,eyeAnalysis_name)
    unique5_output = saveDataFolder + os.sep + 'meanNormedPupilSize_u5_%dmsDelay_%s.npy'%(delay_tb*downsample_ms,eyeAnalysis_name)
    unique6_output = saveDataFolder + os.sep + 'meanNormedPupilSize_u6_%dmsDelay_%s.npy'%(delay_tb*downsample_ms,eyeAnalysis_name)
    ## save file
    np.save(calib_output, pupil_calib_mean)
    np.save(octo_output, pupil_octo_mean)
    np.save(unique1_output, pupil_unique_means[0])
    np.save(unique2_output, pupil_unique_means[1])
    np.save(unique3_output, pupil_unique_means[2])
    np.save(unique4_output, pupil_unique_means[3])
    np.save(unique5_output, pupil_unique_means[4])
    np.save(unique6_output, pupil_unique_means[5])
    # plot scatter plots with regression line
    rval_calib = LumVsPupilSize_ScatterLinRegress(lum_array[0], pupil_calib_mean, 'calib', eyeAnalysis_name, delay_tb*downsample_ms, savePlotsFolder)
    rval_octo = LumVsPupilSize_ScatterLinRegress(lum_array[1], pupil_octo_mean, 'octo', eyeAnalysis_name, delay_tb*downsample_ms, savePlotsFolder)
    rval_u1 = LumVsPupilSize_ScatterLinRegress(lum_array[2], pupil_unique_means[0], 'unique01', eyeAnalysis_name, delay_tb*downsample_ms, savePlotsFolder)
    rval_u2 = LumVsPupilSize_ScatterLinRegress(lum_array[3], pupil_unique_means[1], 'unique02', eyeAnalysis_name, delay_tb*downsample_ms, savePlotsFolder)
    rval_u3 = LumVsPupilSize_ScatterLinRegress(lum_array[4], pupil_unique_means[2], 'unique03', eyeAnalysis_name, delay_tb*downsample_ms, savePlotsFolder)
    rval_u4 = LumVsPupilSize_ScatterLinRegress(lum_array[5], pupil_unique_means[3], 'unique04', eyeAnalysis_name, delay_tb*downsample_ms, savePlotsFolder)
    rval_u5 = LumVsPupilSize_ScatterLinRegress(lum_array[6], pupil_unique_means[4], 'unique05', eyeAnalysis_name, delay_tb*downsample_ms, savePlotsFolder)
    rval_u6 = LumVsPupilSize_ScatterLinRegress(lum_array[7], pupil_unique_means[5], 'unique06', eyeAnalysis_name, delay_tb*downsample_ms, savePlotsFolder)
    # return correlation coefficients
    return [rval_calib, rval_octo, rval_u1, rval_u2, rval_u3, rval_u4, rval_u5, rval_u6]

def normPupilSizeData(pupilSizeArrays_allStim, eyeAnalysis_name):
    normed_pupils = []
    for i, stim_trials in enumerate(pupilSizeArrays_allStim):
        print('Normalizing trials for %s, unique stim %s'%(eyeAnalysis_name, i+1))
        thisUnique_normed = []
        for trial in stim_trials:
            trial_median = np.nanmedian(trial)
            normed_trial = trial/trial_median
            thisUnique_normed.append(normed_trial)
        thisUniqueNormed_array = np.array(thisUnique_normed)
        normed_pupils.append(thisUniqueNormed_array)
    return normed_pupils

def collectRvalsByStimPhase(rvals_allDelays_allPhases):
    rvals_allCalib = []
    rvals_allOcto = []
    rvals_allU1 = []
    rvals_allU2 = []
    rvals_allU3 = []
    rvals_allU4 = []
    rvals_allU5 = []
    rvals_allU6 = []
    for delay in rvals_allDelays_allPhases:
        rvals_allCalib.append(delay[0])
        rvals_allOcto.append(delay[1])
        rvals_allU1.append(delay[2])
        rvals_allU2.append(delay[3])
        rvals_allU3.append(delay[4])
        rvals_allU4.append(delay[5])
        rvals_allU5.append(delay[6])
        rvals_allU6.append(delay[7])
    return [rvals_allCalib, rvals_allOcto, rvals_allU1, rvals_allU2, rvals_allU3, rvals_allU4, rvals_allU5, rvals_allU6]

def drawFitScoresVsDelay(rvals_allPhases, eyeAnalysis_name, downsample_ms, save_folder):
    phase_names = ['calib', 'octo', 'unique1', 'unique2', 'unique3', 'unique4', 'unique5', 'unique6']
    for i, phase in enumerate(rvals_allPhases):
        # figure path and title
        figPath = os.path.join(save_folder, '%s_rValsVsDelays_%s.png'%(phase_names[i], eyeAnalysis_name))
        figTitle = 'Correlation coefficients (r val) vs delays in pupil response time \n Phase: %s, %s'%(phase_names[i], eyeAnalysis_name)
        print('Plotting %s'%(figTitle))
        # draw fit scores vs delay
        plt.figure(figsize=(7, 7), dpi=100)
        plt.suptitle(figTitle, fontsize=12, y=0.98)
        plt.xlabel('Delay of pupil size data (ms)')
        plt.ylabel('Correlation coefficient')
        plt.xticks(np.arange(15), np.arange(15)*downsample_ms)
        plt.plot(phase, 'g')
        # save figure and close
        plt.savefig(figPath)
        plt.close()


###################################
# DATA AND OUTPUT FILE LOCATIONS
###################################
# List relevant data locations: these are for laptop
#root_folder = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\dataPythonWorkflows"
#plots_folder = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\plots"
# List relevant data locations: these are for office desktop
root_folder = r"C:\Users\Kampff_Lab\Dropbox\SurprisingMinds\analysis\dataPythonWorkflows"
plots_folder = r"C:\Users\Kampff_Lab\Dropbox\SurprisingMinds\analysis\plots"
# set up folders
# world camera average luminance csv files
worldCamLum_folder = os.path.join(root_folder, 'worldLums')
# scatter plot output folder
pupilSize_folder = os.path.join(plots_folder, "pupilSizeAnalysis")
Rco_folder = os.path.join(pupilSize_folder, 'rightContours')
Rci_folder = os.path.join(pupilSize_folder, 'rightCircles')
Lco_folder = os.path.join(pupilSize_folder, 'leftContours')
Lci_folder = os.path.join(pupilSize_folder, 'leftCircles')
# normed mean pupil sizes output folder
normedMeanPupilSizes_folder = os.path.join(root_folder, 'normedMeanPupilSizes')
# Create folders if they do not exist
output_folders = [pupilSize_folder, Rco_folder, Rci_folder, Lco_folder, Lci_folder, normedMeanPupilSizes_folder]
for output_folder in output_folders:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

###################################
# TIMING/SAMPLING VARIABLES FOR DATA EXTRACTION
###################################
# downsample = collect data from every 40ms or other multiples of 20
downsampled_bucket_size_ms = 40
original_bucket_size_in_ms = 4
max_length_of_stim_vid = 60000 # milliseconds
no_of_time_buckets = max_length_of_stim_vid/original_bucket_size_in_ms
downsampled_no_of_time_buckets = max_length_of_stim_vid/downsampled_bucket_size_ms
new_time_bucket_sample_rate = downsampled_bucket_size_ms/original_bucket_size_in_ms
milliseconds_for_baseline = 3000
baseline_no_buckets = int(milliseconds_for_baseline/new_time_bucket_sample_rate)

###################################
# STIMULI VID INFO
###################################
stim_vids = [24.0, 25.0, 26.0, 27.0, 28.0, 29.0]
stim_name_to_float = {"Stimuli24": 24.0, "Stimuli25": 25.0, "Stimuli26": 26.0, "Stimuli27": 27.0, "Stimuli28": 28.0, "Stimuli29": 29.0}
stim_float_to_name = {24.0: "Stimuli24", 25.0: "Stimuli25", 26.0: "Stimuli26", 27.0: "Stimuli27", 28.0: "Stimuli28", 29.0: "Stimuli29"}

###################################
# BEGIN PUPIL DATA EXTRACTION 
###################################
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
activation_count = {}
analysed_count = {}
stimuli_tbucketed = {key:[] for key in stim_vids}
# consolidate csv files from multiple days into one data structure
day_folders = sorted(os.listdir(root_folder))
# find pupil data on dropbox
pupil_folders = fnmatch.filter(day_folders, 'SurprisingMinds_*')
# first day was a debugging session, so skip it
pupil_folders = pupil_folders[1:]

### WHILE DEBUGGING ###
#pupil_folders = pupil_folders[5:10]
# if currently still running pupil finding analysis...
#pupil_folders = pupil_folders[:-1]
#### --------------- ####

# collect dates for which pupil extraction fails
failed_days = []
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

        analysed_count[day_name] = [num_good_right_trials, num_good_left_trials]
        activation_count[day_name] = [num_right_activations, num_left_activations]
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

        # append position data to global data structure
        for i in range(len(all_position_X_data)):
            for stimulus in all_position_X_data[i]:
                for index in range(len(all_position_X_data[i][stimulus])):
                    all_trials_position_X_data[i][stimulus].append(all_position_X_data[i][stimulus][index])
        for i in range(len(all_position_Y_data)):
            for stimulus in all_position_Y_data[i]:
                for index in range(len(all_position_Y_data[i][stimulus])):
                    all_trials_position_Y_data[i][stimulus].append(all_position_Y_data[i][stimulus][index])
        # append size data to global data structure
        for i in range(len(all_size_data)):
            for stimulus in all_size_data[i]:
                for index in range(len(all_size_data[i][stimulus])):
                    all_trials_size_data[i][stimulus].append(all_size_data[i][stimulus][index])
        print("Day {day} succeeded!".format(day=day_name))
    except Exception:
        failed_days.append(day_name)
        print("Day {day} failed!".format(day=day_name))

# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
###################################
# Load world camera luminance files
###################################
worldCamLum_files = glob.glob(worldCamLum_folder + os.sep + '*.npy')
avgLum_allPhases = []
phaseOrder = []
uniqueLens = []
for lum_file in worldCamLum_files:
    # get stimuli phase type
    phase_type = os.path.basename(lum_file).split('_')[0]
    avgLum = np.load(lum_file)
    avgLum_allPhases.append(avgLum)
    phaseOrder.append(phase_type)
    lumLen = int(os.path.basename(lum_file).split('_')[2].split('.')[0][:-3])
    if phase_type == 'meanCalib':
        calibLen = lumLen
        continue
    if phase_type == 'meanOcto':
        octoLen = lumLen
        continue
    else:
        uniqueLens.append(lumLen)

###################################
# Normalize pupil size data 
###################################
# Right Contours
R_contours_allStim = [all_trials_size_data[0][24.0], all_trials_size_data[0][25.0], all_trials_size_data[0][26.0], all_trials_size_data[0][27.0], all_trials_size_data[0][28.0], all_trials_size_data[0][29.0]]
Rco_normed = normPupilSizeData(R_contours_allStim, 'right contours')
# Right Circles
R_circles_allStim = [all_trials_size_data[1][24.0], all_trials_size_data[1][25.0], all_trials_size_data[1][26.0], all_trials_size_data[1][27.0], all_trials_size_data[1][28.0], all_trials_size_data[1][29.0]]
Rci_normed = normPupilSizeData(R_circles_allStim, 'right circles')
# Left Contours
L_contours_allStim = [all_trials_size_data[2][24.0], all_trials_size_data[2][25.0], all_trials_size_data[2][26.0], all_trials_size_data[2][27.0], all_trials_size_data[2][28.0], all_trials_size_data[2][29.0]]
Lco_normed = normPupilSizeData(L_contours_allStim, 'left contours')
# Left Circles
L_circles_allStim = [all_trials_size_data[3][24.0], all_trials_size_data[3][25.0], all_trials_size_data[3][26.0], all_trials_size_data[3][27.0], all_trials_size_data[3][28.0], all_trials_size_data[3][29.0]]
Lci_normed = normPupilSizeData(L_circles_allStim, 'left circles')

###################################
# split normed pupil size arrays based on different delays of pupil reaction
# save split normed pupil size arrays as binary files
# create scatter plot of pupil size against world cam luminance values
# include least squares regression line in scatter plot
###################################
delays = range(15)
rvals_Rco_allDelays = []
rvals_Rci_allDelays = []
rvals_Lco_allDelays = []
rvals_Lci_allDelays = []
for delay in delays:
    print('Delay: %d timebucket(s)'%(delay))
    rvals_Rco = splitPupils_withDelay_plotScatterLinRegress(delay, downsampled_bucket_size_ms, avgLum_allPhases, Rco_normed, calibLen, uniqueLens, octoLen, 'RightContour', Rco_folder, normedMeanPupilSizes_folder)
    rvals_Rco_allDelays.append(rvals_Rco)
    rvals_Rci = splitPupils_withDelay_plotScatterLinRegress(delay, downsampled_bucket_size_ms, avgLum_allPhases, Rci_normed, calibLen, uniqueLens, octoLen, 'RightCircles', Rci_folder, normedMeanPupilSizes_folder)
    rvals_Rci_allDelays.append(rvals_Rci)
    rvals_Lco = splitPupils_withDelay_plotScatterLinRegress(delay, downsampled_bucket_size_ms, avgLum_allPhases, Lco_normed, calibLen, uniqueLens, octoLen, 'LeftContour', Lco_folder, normedMeanPupilSizes_folder)
    rvals_Lco_allDelays.append(rvals_Lco)
    rvals_Lci = splitPupils_withDelay_plotScatterLinRegress(delay, downsampled_bucket_size_ms, avgLum_allPhases, Lci_normed, calibLen, uniqueLens, octoLen, 'LeftCircles', Lci_folder, normedMeanPupilSizes_folder)
    rvals_Lci_allDelays.append(rvals_Lci)

###################################
# plot fit scores (rvals) vs delay
# for each delay, rvals = [rval_calib, rval_octo, rval_u1, rval_u2, rval_u3, rval_u4, rval_u5, rval_u6]
###################################
# collect rvals by stimulus phase
rvals_Rco_allPhases = collectRvalsByStimPhase(rvals_Rco_allDelays)
rvals_Rci_allPhases = collectRvalsByStimPhase(rvals_Rci_allDelays)
rvals_Lco_allPhases = collectRvalsByStimPhase(rvals_Lco_allDelays)
rvals_Lci_allPhases = collectRvalsByStimPhase(rvals_Lci_allDelays)
# plot fit scores vs delay for each phase
drawFitScoresVsDelay(rvals_Rco_allPhases, 'RightContours', downsampled_bucket_size_ms, Rco_folder)
drawFitScoresVsDelay(rvals_Rci_allPhases, 'RightCircles', downsampled_bucket_size_ms, Rci_folder)
drawFitScoresVsDelay(rvals_Lco_allPhases, 'LeftContours', downsampled_bucket_size_ms, Lco_folder)
drawFitScoresVsDelay(rvals_Lci_allPhases, 'LeftCircles', downsampled_bucket_size_ms, Lci_folder)

# FIN