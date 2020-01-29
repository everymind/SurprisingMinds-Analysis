### ------------------------------------------------------------------------- ###
### Create CSV files with average luminance per frame of world camera vids during
### trials, categorized by calibration, octopus, and unique sequences.
### ------------------------------------------------------------------------- ###
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
# List relevant data locations: this is for laptop
#root_folder = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\dataPythonWorkflows"
# List relevant data locations: this is for office desktop (windows)
root_folder = r"C:\Users\Kampff_Lab\Dropbox\SurprisingMinds\analysis\dataPythonWorkflows"
# set up folders
worldVid_lums_folder = os.path.join(root_folder, "worldLums")
# Create folders they do not exist
if not os.path.exists(worldVid_lums_folder):
    os.makedirs(worldVid_lums_folder)

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
allMonths_meanWorldVidArrays = {}
for unique_stim in stim_vids:
    allMonths_meanWorldVidArrays[unique_stim] = {}
    allMonths_meanWorldVidArrays[unique_stim]['Vid Count'] = 0
### EXTRACT, UNRAVEL, SAVE TO FILE TIME BINNED STIM VIDEOS ###
# update list of completed world vid average folders on dropbox
day_folders = sorted(os.listdir(root_folder))
avg_world_vid_folders = fnmatch.filter(day_folders, 'WorldVidAverage_*')
updated_folders_to_extract = []
for avg_world_vid_folder in avg_world_vid_folders:
    folder_year_month = avg_world_vid_folder.split('_')[1]
    if folder_year_month not in allMonths_meanWorldVidArrays.keys():
        updated_folders_to_extract.append(avg_world_vid_folder)

#### WHILE DEBUGGING ####
#updated_folders_to_extract = updated_folders_to_extract[4:6]
#### --------------- ####

# extract, unravel, calculate mean luminance of each frame, create array of mean luminances for each stim type
for month_folder in updated_folders_to_extract:
    month_name = month_folder.split('_')[1]
    month_folder_path = os.path.join(root_folder, month_folder)
    # unravel
    unraveled_monthly_world_vids = load_avg_world_unraveled(month_folder_path)
    # downsample
    print("Downsampling monthly averaged stimulus videos for {month}".format(month=month_name))
    downsampled_monthly_world_vids = downsample_avg_world_vids(unraveled_monthly_world_vids, original_bucket_size_in_ms, downsampled_bucket_size_ms)
    # now need to convert these frame arrays into luminance value, one per timebucket
    for unique_stim in downsampled_monthly_world_vids:
        thisMonth_thisStim_frames = downsampled_monthly_world_vids[unique_stim]
        thisMonth_thisStim_lums = []
        for key in thisMonth_thisStim_frames:
            if key == 'Vid Count':
                allMonths_meanWorldVidArrays[unique_stim]['Vid Count'] = allMonths_meanWorldVidArrays[unique_stim]['Vid Count'] + thisMonth_thisStim_frames['Vid Count']
                continue
            if key == 'Vid Dimensions':
                continue
            else:
                frame = thisMonth_thisStim_frames[key]
                lum = np.mean(frame[:])
                thisMonth_thisStim_lums.append(lum)
        thisMonth_thisStim_lums_array = np.array(thisMonth_thisStim_lums)
        allMonths_meanWorldVidArrays[unique_stim][month_name] = thisMonth_thisStim_lums_array
### END MONTHLY AVERAGE DATA EXTRACTION ###
###
### AVERAGE ACROSS ALL MONTHS ###
for unique_stim in allMonths_meanWorldVidArrays:
    allMonthlyMeans = []
    shortest = 2000
    for key in allMonths_meanWorldVidArrays[unique_stim]:
        if key == 'Vid Count':
            continue
        else:
            thisMonthMean = allMonths_meanWorldVidArrays[unique_stim][key]
            if len(thisMonthMean)<shortest:
                shortest = len(thisMonthMean)
            allMonthlyMeans.append(thisMonthMean)       
    # make all arrays same length
    allMonthlyMeans_truncated = []
    for monthlyMean in allMonthlyMeans:
        monthlyMean_truncated = monthlyMean[:shortest]
        allMonthlyMeans_truncated.append(monthlyMean_truncated)
    allMonthlyMeans_array = np.array(allMonthlyMeans_truncated)
    thisStimMeanLum = np.nanmean(allMonthlyMeans_array, axis=0)
    allMonths_meanWorldVidArrays[unique_stim]['All Months'] = thisStimMeanLum

### SPLIT ARRAYS INTO CALIB, OCTO, AND UNIQUE PHASES ###
# Moments of interest for each stimulus type
all_avg_world_moments = {}
# Stimulus 24.0
all_avg_world_moments[24.0] = {'calibration start': {0:['2017-10','2018-05']},
'do not move your head': {3:['2017-10','2018-05']},
'upper left dot appears': {102:['2017-10','2017-11','2018-03']},
'lower right dot appears': {170:['2017-10','2018-05']},
'lower left dot appears': {238:['2017-10','2018-05']},
'upper right dot appears': {306:['2017-10','2018-05']},
'center dot appears': {374:['2017-10','2018-05']},
'calibration end': {441:['2017-10','2017-11','2018-03']},
'unique start': {442:['2017-10','2018-03','2018-05'],443:['2017-11']},
'cat appears': {463:['2017-10','2018-01','2018-05'], 464:['2017-11']},
'cat front paws visible': {473:['2017-10','2018-01','2018-05'], 474:['2017-11']},
'cat lands on toy': {513:['2017-10'], 514:['2018-05']},
'cat back paws bounce': {549:['2017-10'],547:['2018-05']},
'unique end': {596:['2017-10','2017-11'],598:['2018-03']},
'octo start': {595:['2017-10','2018-03'],596:['2017-11']},
'fish turns': {645:['2017-10','2018-05']},
'octopus fully decamouflaged': {766:['2018-05'], 767:['2017-10']},
'camera zooms in on octopus': {860:['2017-10','2018-05']},
'octopus inks': {882:['2017-10'],883:['2017-11','2018-03']},
'camera clears ink cloud': {916:['2017-10'],920:['2018-05']},
'octo end': {987:['2017-10'],989:['2017-11'],990:['2018-03']}}
# Stimulus 25.0
all_avg_world_moments[25.0] = {'calibration start': {0:['2017-10','2017-11','2018-03']},
'do not move your head': {3:['2017-10','2018-05']},
'upper left dot appears': {102:['2017-10','2017-11','2018-03']},
'lower right dot appears': {170:['2017-10','2018-05']},
'lower left dot appears': {239:['2017-10'],238:['2018-05']},
'upper right dot appears': {307:['2017-10'],306:['2018-05']},
'center dot appears': {375:['2017-10'],374:['2018-05']},
'calibration end': {441:['2017-10','2017-11','2018-03']},
'unique start': {442:['2018-03'],443:['2017-10','2017-11']},
'fingers appear': {443:['2017-10'], 442:['2018-05']},
'bird flies towards fingers': {462:['2018-05'],463:['2017-10']},
'beak contacts food': {491:['2017-10'],492:['2018-05']},
'wings at top of frame': {535:['2017-10','2018-05']},
'bird flutters': {553:['2017-10'], 553:['2018-05']},
'bird lands': {561:['2017-10'], 562:['2018-05']},
'bird flies past fingers': {573:['2017-10','2018-05']},
'unique end': {599:['2017-10'],600:['2017-11'],601:['2018-03']},
'octo start': {599:['2017-10','2017-11','2018-03']},
'fish turns': {649:['2017-10','2018-05']},
'octopus fully decamouflaged': {770:['2017-10','2018-05']},
'camera zooms in on octopus': {863:['2018-05'],864:['2017-10']},
'octopus inks': {885:['2017-10','2018-03'],886:['2017-11']},
'camera clears ink cloud': {919:['2017-10'],923:['2018-05']},
'octo end': {989:['2017-10'],993:['2017-11'],994:['2018-03']}}
# Stimulus 26.0
all_avg_world_moments[26.0] = {'calibration start': {0:['2017-10','2017-11','2018-03']},
'do not move your head': {2:['2018-05'],3:['2017-10']},
'upper left dot appears': {102:['2017-10','2017-11','2018-03']},
'lower right dot appears': {170:['2017-10','2018-05']},
'lower left dot appears': {238:['2017-10','2018-05']},
'upper right dot appears': {306:['2017-10','2018-05']},
'center dot appears': {374:['2017-10','2018-05']},
'calibration end': {441:['2017-10','2017-11','2018-03']},
'unique start': {442:['2017-10','2018-03'],443:['2017-11']},
'eyespots appear': {449:['2017-10', '2018-05']},
'eyespots disappear, eyes darken': {487:['2017-10','2018-05']},
'arms spread': {533:['2017-10'], 534:['2018-05']},
'arms in, speckled mantle': {558:['2017-10'], 561:['2018-05']},
'unique end': {663:['2017-10'],665:['2017-11','2018-03']},
'octo start': {662:['2017-10'],663:['2018-03'],664:['2017-11']},
'fish turns': {712:['2017-10','2018-05']},
'octopus fully decamouflaged': {833:['2017-10','2018-05']},
'camera zooms in on octopus': {927:['2017-10','2018-05']},
'octopus inks': {949:['2017-10'],951:['2017-11','2018-03']},
'camera clears ink cloud': {983:['2017-10'],987:['2018-05']},
'octo end': {1054:['2017-10'],1059:['2017-11','2018-03']}}
# Stimulus 27.0
all_avg_world_moments[27.0] = {'calibration start': {0:['2017-10','2017-11','2018-03']},
'do not move your head': {3:['2017-10','2018-05']},
'upper left dot appears': {102:['2017-10','2017-11','2018-03']},
'lower right dot appears': {170:['2017-10','2018-05']},
'lower left dot appears': {238:['2017-10','2018-05']},
'upper right dot appears': {306:['2018-05'],307:['2017-10']},
'center dot appears': {374:['2018-05'],375:['2017-10']},
'calibration end': {441:['2017-10','2017-11','2018-03']},
'unique start': {443:['2017-10','2017-11','2018-03']},
'cuttlefish appears': {443:['2017-10','2018-05']},
'tentacles go ballistic': {530:['2017-10','2018-05']},
'unique end': {606:['2017-10'],607:['2017-11','2018-03']},
'octo start': {605:['2017-10','2017-11'],606:['2018-03']},
'fish turns': {655:['2017-10','2018-05']},
'octopus fully decamouflaged': {776:['2017-10','2018-05']},
'camera zooms in on octopus': {869:['2018-05'],870:['2017-10']},
'octopus inks': {892:['2017-10'],893:['2017-11','2018-03']},
'camera clears ink cloud': {926:['2017-10'],929:['2018-05']},
'octo end': {996:['2017-10'],1000:['2017-11','2018-03']}}
# Stimulus 28.0
all_avg_world_moments[28.0] = {'calibration start': {0:['2017-10','2017-11','2018-03']},
'do not move your head': {2:['2018-05'],3:['2017-10']},
'upper left dot appears': {102:['2017-10','2017-11','2018-03']},
'lower right dot appears': {170:['2017-10','2018-05']},
'lower left dot appears': {238:['2017-10','2018-05']},
'upper right dot appears': {306:['2017-10','2018-05']},
'center dot appears': {374:['2018-05'],375:['2017-10']},
'calibration end': {441:['2017-10','2017-11','2018-03']},
'unique start': {442:['2018-03'],443:['2017-10','2017-11']},
'fish scatter': {456:['2017-10','2018-04','2018-10']},
'center fish turns': {469:['2017-10'], 470:['2018-04'], 471:['2018-10']},
'center fish swims to left': {494:['2018-04','2018-10'], 495:['2017-10']},
'camera clears red ferns': {503:['2017-10'],506:['2018-04'],509:['2018-10']},
'unique end': {662:['2017-10'],663:['2017-11'],666:['2018-03']},
'octo start': {661:['2017-10'],662:['2018-03'],663:['2017-11']},
'fish turns': {711:['2017-10','2018-05']},
'octopus fully decamouflaged': {832:['2017-10'],834:['2018-05']},
'camera zooms in on octopus': {927:['2017-10','2018-05']},
'octopus inks': {948:['2017-10'],950:['2017-11','2018-03']},
'camera clears ink cloud': {982:['2017-10'],986:['2018-05']},
'octo end': {1054:['2017-10'],1056:['2017-11'],1059:['2018-03']}}
# Stimulus 29.0
all_avg_world_moments[29.0] = {'calibration start': {0:['2017-10','2017-11','2018-03']},
'do not move your head': {3:['2017-10','2018-05']},
'upper left dot appears': {102:['2017-10','2017-11','2018-03']},
'lower right dot appears': {170:['2017-10','2018-05']},
'lower left dot appears': {238:['2017-10','2018-05']},
'upper right dot appears': {306:['2017-10','2018-05']},
'center dot appears': {374:['2017-10','2018-05']},
'calibration end': {441:['2017-10','2017-11','2018-03']},
'unique start': {442:['2017-10'],443:['2017-11','2018-03']},
'fish 1 appears': {457:['2017-10','2018-05']},
'fish 1 turns': {495:['2017-10','2018-05']}, 
'fish 2 appears': {538:['2017-10','2018-05']},
'fish 2 touches mirror image': {646:['2017-10','2018-05']},
'fish 2 disappears': {661:['2017-10','2018-05']}, 
'fish 1 touches mirror image': {685:['2017-10','2018-05']},
'fish 1 disappears': {702:['2017-10','2018-05']}, 
'unique end': {717:['2017-10','2017-11'],718:['2018-03']},
'octo start': {716:['2017-10','2018-03'],717:['2017-11']},
'fish turns': {766:['2017-10','2018-03']},
'octopus fully decamouflaged': {887:['2017-10','2018-05']},
'camera zooms in on octopus': {981:['2017-10','2018-05']},
'octopus inks': {1003:['2017-10'],1004:['2017-11','2018-03']},
'camera clears ink cloud': {1037:['2017-10'],1041:['2018-05']},
'octo end': {1108:['2017-10'],1110:['2017-11'],1112:['2018-03']}}
# split world vid lum arrays
allCalib = []
allOcto = []
allUnique = []
calibLens = []
octoLens = []
uniqueLens = []
uniqueOrder = []
shortestCalib = 2000
shortestOcto = 2000
# cut out each phase of the stimuli
for unique_stim in allMonths_meanWorldVidArrays:
    fullMeanWorldVid = allMonths_meanWorldVidArrays[unique_stim]['All Months']
    ## CALIB
    calibStart = []
    for key in all_avg_world_moments[unique_stim]['calibration start']:
        calibStart.append(key)
    calibStart_tb = np.min(calibStart)
    calibEnd = []
    for key in all_avg_world_moments[unique_stim]['calibration end']:
        calibEnd.append(key)
    calibEnd_tb = np.max(calibEnd)
    # cut out calib phase from full world vid lum array
    thisStim_meanCalib = fullMeanWorldVid[calibStart_tb:calibEnd_tb]
    if len(thisStim_meanCalib)<shortestCalib:
        shortestCalib = len(thisStim_meanCalib)
    allCalib.append(thisStim_meanCalib)
    calibLen = calibEnd_tb - calibStart_tb
    calibLens.append(calibLen)
    ## OCTO
    octoStart = []
    for key in all_avg_world_moments[unique_stim]['octo start']:
        octoStart.append(key)
    octoStart_tb = np.min(octoStart)
    octoEnd = []
    for key in all_avg_world_moments[unique_stim]['octo end']:
        octoEnd.append(key)
    octoEnd_tb = np.max(octoEnd)
    # cut out octo phase from full world vid lum array
    thisStim_meanOcto = fullMeanWorldVid[octoStart_tb:octoEnd_tb]
    if len(thisStim_meanOcto)<shortestOcto:
        shortestOcto = len(thisStim_meanOcto)
    allOcto.append(thisStim_meanOcto)
    octoLen = octoEnd_tb - octoStart_tb
    octoLens.append(octoLen)
    ### UNIQUE
    thisUniqueStart = []
    for key in all_avg_world_moments[unique_stim]['unique start']:
        thisUniqueStart.append(key)
    thisUniqueStart_tb = np.min(thisUniqueStart)
    thisUniqueEnd = []
    for key in all_avg_world_moments[unique_stim]['unique end']:
        thisUniqueEnd.append(key)
    thisUniqueEnd_tb = np.max(thisUniqueEnd)
    uniqueLen = thisUniqueEnd_tb - thisUniqueStart_tb
    uniqueLens.append(uniqueLen)
    # cut out unique phase from full world vid lum array
    thisStim_meanUnique = fullMeanWorldVid[thisUniqueStart_tb:thisUniqueEnd_tb]
    allUnique.append(thisStim_meanUnique)
    uniqueOrder.append(unique_stim)
# average calib and octo across all stimuli
allCalib_truncated = []
for calib in allCalib:
    calib_truncated = calib[:shortestCalib]
    allCalib_truncated.append(calib_truncated)
allCalib_array = np.array(allCalib_truncated)
allCalib_mean = np.mean(allCalib_array, axis=0)
allOcto_truncated = []
for octo in allOcto:
    octo_truncated = octo[:shortestOcto]
    allOcto_truncated.append(octo_truncated)
allOcto_array = np.array(allOcto_truncated)
allOcto_mean = np.mean(allOcto_array, axis=0)

### SAVE AS CSV FILES ###
# generate file names to include number of videos that went into the mean lum array
totalVidCount = 0
for unique_stim in allMonths_meanWorldVidArrays:
    totalVidCount = totalVidCount + allMonths_meanWorldVidArrays[unique_stim]['Vid Count']
# filepaths
calib_output = worldVid_lums_folder + os.sep + 'meanCalib_%sVids_%dTBs.data' % (totalVidCount, max(calibLens))
octo_output = worldVid_lums_folder + os.sep + 'meanOcto_%sVids_%dTBs.data' % (totalVidCount, max(octoLens))
unique24_output = worldVid_lums_folder + os.sep + 'meanUnique01_%sVids_%dTBs.data' % (allMonths_meanWorldVidArrays[24.0]['Vid Count'], uniqueLens[uniqueOrder.index(24.0)])
unique25_output = worldVid_lums_folder + os.sep + 'meanUnique02_%sVids_%dTBs.data' % (allMonths_meanWorldVidArrays[25.0]['Vid Count'], uniqueLens[uniqueOrder.index(25.0)])
unique26_output = worldVid_lums_folder + os.sep + 'meanUnique03_%sVids_%dTBs.data' % (allMonths_meanWorldVidArrays[26.0]['Vid Count'], uniqueLens[uniqueOrder.index(26.0)])
unique27_output = worldVid_lums_folder + os.sep + 'meanUnique04_%sVids_%dTBs.data' % (allMonths_meanWorldVidArrays[27.0]['Vid Count'], uniqueLens[uniqueOrder.index(27.0)])
unique28_output = worldVid_lums_folder + os.sep + 'meanUnique05_%sVids_%dTBs.data' % (allMonths_meanWorldVidArrays[28.0]['Vid Count'], uniqueLens[uniqueOrder.index(28.0)])
unique29_output = worldVid_lums_folder + os.sep + 'meanUnique06_%sVids_%dTBs.data' % (allMonths_meanWorldVidArrays[29.0]['Vid Count'], uniqueLens[uniqueOrder.index(29.0)])
# save to file
allCalib_mean.tofile(calib_output)
allOcto_mean.tofile(octo_output)
allUnique[0].tofile(unique24_output)
allUnique[1].tofile(unique25_output)
allUnique[2].tofile(unique26_output)
allUnique[3].tofile(unique27_output)
allUnique[4].tofile(unique28_output)
allUnique[5].tofile(unique29_output)

# FIN
