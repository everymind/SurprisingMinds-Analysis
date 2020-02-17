### ------------------------------------------------------------------------- ###
### Create binary files of raw stim vid luminance values fitted to world cam stim vid presentation timings
### use world camera vids for timing, use raw vid luminance values extracted via bonsai
### also save world cam luminance as sanity check/ground truth 
### create monthly averages of both raw live stim vid and world cam sanity check
### output as data files 
### ------------------------------------------------------------------------- ###
import os
import glob
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import zipfile
import shutil
import fnmatch
import sys
import math
import csv
###################################
# FUNCTIONS
###################################
def unpack_to_temp(path_to_zipped, path_to_temp):
    try:
        # copy zip file to current working directory
        #print("Copying {folder} to current working directory...".format(folder=path_to_zipped))
        current_working_directory = os.getcwd()
        copied_zipped = shutil.copy2(path_to_zipped, current_working_directory)
        path_to_copied_zipped = os.path.join(current_working_directory, copied_zipped.split(sep=os.sep)[-1])
        # unzip the folder
        #print("Unzipping files in {folder}...".format(folder=path_to_copied_zipped))
        day_unzipped = zipfile.ZipFile(path_to_copied_zipped, mode="r")
        # extract files into temp folder
        day_unzipped.extractall(path_to_temp)
        # close the unzipped file
        day_unzipped.close()
        #print("Finished unzipping {folder}!".format(folder=path_to_copied_zipped))
        # destroy copied zipped file
        #print("Deleting {file}...".format(file=path_to_copied_zipped))
        os.remove(path_to_copied_zipped)
        #print("Deleted {file}!".format(file=path_to_copied_zipped))
        return True
    except Exception: 
        print("Could not unzip {folder}".format(folder=path_to_zipped))    
        return False

def list_sub_folders(path_to_root_folder):
    # List all sub folders
    sub_folders = []
    for folder in os.listdir(path_to_root_folder):
        if(os.path.isdir(os.path.join(path_to_root_folder, folder))):
            sub_folders.append(os.path.join(path_to_root_folder, folder))
    return sub_folders

def make_time_buckets(start_timestamp, bucket_size_ms, end_timestamp, fill_pattern): 
    start_timestamp = start_timestamp.split('+')[0][:-3]
    end_timestamp = end_timestamp.split('+')[0][:-3]
    buckets_start_time = datetime.datetime.strptime(start_timestamp, "%Y-%m-%dT%H:%M:%S.%f")
    buckets_end_time = datetime.datetime.strptime(end_timestamp, "%Y-%m-%dT%H:%M:%S.%f")

    current_bucket = buckets_start_time
    time_buckets = []
    window = datetime.timedelta(milliseconds=bucket_size_ms)
    while current_bucket <= buckets_end_time:
        time_buckets.append(current_bucket)
        current_bucket = current_bucket + window

    bucket_list = dict.fromkeys(time_buckets)

    for key in time_buckets: 
        bucket_list[key] = fill_pattern

    return bucket_list

def find_nearest_timestamp_key(timestamp_to_check, dict_of_timestamps, time_window):
    for key in dict_of_timestamps.keys():
        if key <= timestamp_to_check <= (key + time_window):
            return key

def supersampled_worldCam_rawLiveVid(video_path, video_timestamps, rawStimVidData_dict, output_folder, bucket_size_ms):
    # Get video file details
    video_name = video_path.split(os.sep)[-1]
    video_date = video_name.split('_')[0]
    video_time = video_name.split('_')[1]
    video_stim_number = video_name.split('_')[2]
    # Open world video
    world_vid = cv2.VideoCapture(video_path)
    vid_width = int(world_vid.get(3))
    vid_height = int(world_vid.get(4))
    # create rawLiveVid output array
    first_timestamp = video_timestamps[0]
    last_timestamp = video_timestamps[-1]
    rawLiveVid_initializePattern = np.nan
    rawLiveVid_buckets = make_time_buckets(first_timestamp, bucket_size_ms, last_timestamp, rawLiveVid_initializePattern)
    sanityCheck_initializePattern = np.empty((vid_height*vid_width,))
    sanityCheck_initializePattern[:] = np.nan
    worldCam_sanityCheck_buckets = make_time_buckets(first_timestamp, bucket_size_ms, last_timestamp, sanityCheck_initializePattern)
    # Loop through 4ms time buckets of world video to find nearest frame and save 2-d matrix of pixel values in that frame
    # stimStructure = ['DoNotMove-English', 'Calibration', 'stimuli024', 'stimuli025', 'stimuli026', 'stimuli027', 'stimuli028', 'stimuli029', ]
    doNotMove_frameCount = rawStimVidData_dict['DoNotMove-English']['Number of Frames']
    calib_frameCount = rawStimVidData_dict['Calibration']['Number of Frames']
    # keep track of how many frames have been processed
    frame_count = 0
    for timestamp in video_timestamps:
        # find the time bucket into which this frame falls
        timestamp = timestamp.split('+')[0][:-3]
        timestamp_dt = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f")
        bucket_window = datetime.timedelta(milliseconds=bucket_size_ms)
        # fill in luminance values from world cam video as a sanity check
        currentKey_sanityCheck = find_nearest_timestamp_key(timestamp_dt, worldCam_sanityCheck_buckets, bucket_window)
        # Read frame at current position
        # should this be at current key??
        ret, frame = world_vid.read()
        # Make sure the frame exists!
        if frame is not None:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # flatten the frame into a list
            flattened_gray = gray.ravel()
            flattened_gray = flattened_gray.astype(None)
            # append to dictionary stim_buckets
            worldCam_sanityCheck_buckets[currentKey_sanityCheck] = flattened_gray
        # fill in luminance values from raw videos based on timing of framerate in world camera timestamps
        currentKey_rLV = find_nearest_timestamp_key(timestamp_dt, rawLiveVid_buckets, bucket_window)
        if frame_count < doNotMove_frameCount:
            rawVidPhase = 'DoNotMove-English'
            frame_index = frame_count
        if doNotMove_frameCount <= frame_count < doNotMove_frameCount + calib_frameCount:
            rawVidPhase = 'Calibration'
            frame_index = frame_count - doNotMove_frameCount
        if doNotMove_frameCount + calib_frameCount <= frame_count:
            rawVidPhase = video_stim_number
            if frame_count < doNotMove_frameCount + calib_frameCount + rawStimVidData_dict[rawVidPhase]['Number of Frames']:
                frame_index = frame_count - doNotMove_frameCount - calib_frameCount
            else:
                break
        rawLiveVid_buckets[currentKey_rLV] = rawStimVidData_dict[rawVidPhase]['Luminance per Frame'][frame_index]
        #print('Processing frame %d from %s phase (total frame count: %d)' % (frame_index, rawVidPhase, frame_count))
        frame_count = frame_count + 1
    # release video capture
    world_vid.release()
    # generate rawLiveVid luminance array output
    supersampled_rawLiveVid = []
    current_lumVal = 0
    for timestamp in sorted(rawLiveVid_buckets.keys()):
        if rawLiveVid_buckets[timestamp] is not np.nan:
            supersampled_rawLiveVid.append(rawLiveVid_buckets[timestamp])
            current_lumVal = rawLiveVid_buckets[timestamp]
        else:
            supersampled_rawLiveVid.append(current_lumVal)
    supersampled_rawLiveVid_array = np.array(supersampled_rawLiveVid)
    # generate worldCam sanityCheck luminance array output
    supersampled_worldCam = []
    current_frame = sanityCheck_initializePattern
    for timestamp in sorted(worldCam_sanityCheck_buckets.keys()):
        if worldCam_sanityCheck_buckets[timestamp] is not np.nan:
            supersampled_worldCam.append(worldCam_sanityCheck_buckets[timestamp])
            current_frame = worldCam_sanityCheck_buckets[timestamp]
        else:
            supersampled_worldCam.append(current_frame)
    supersampled_worldCam_array = np.array(supersampled_worldCam)
    # return worldCam sanity check
    return vid_width, vid_height, supersampled_worldCam_array, supersampled_rawLiveVid_array

def add_to_daily_worldCam_dict(this_trial_world_vid_frames, this_trial_stim_num, daily_world_vid_dict):
    # keep track of how many videos are going into the average for this stim
    daily_world_vid_dict[this_trial_stim_num]['Vid Count'] = daily_world_vid_dict[this_trial_stim_num].get('Vid Count', 0) + 1
    this_trial_stim_vid = {}
    for tb, row in enumerate(this_trial_world_vid_frames):
        tbucket_num = tb
        flattened_frame = row
        this_trial_stim_vid[tbucket_num] = flattened_frame
    for tbucket in this_trial_stim_vid.keys():
        if tbucket in daily_world_vid_dict[this_trial_stim_num].keys():
            daily_world_vid_dict[this_trial_stim_num][tbucket]['Trial Count'] = daily_world_vid_dict[this_trial_stim_num][tbucket]['Trial Count'] + 1
            daily_world_vid_dict[this_trial_stim_num][tbucket]['Summed Frame'] = daily_world_vid_dict[this_trial_stim_num][tbucket]['Summed Frame'] + this_trial_stim_vid[tbucket]
        else: 
            daily_world_vid_dict[this_trial_stim_num][tbucket] = {'Trial Count': 1, 'Summed Frame': this_trial_stim_vid[tbucket]}

def add_to_daily_rawLiveVid_dict(this_trial_rawLive_vid_frames, this_trial_stim_num, daily_rawLive_vid_dict):
    # keep track of how many videos are going into the average for this stim
    daily_rawLive_vid_dict[this_trial_stim_num]['Vid Count'] = daily_rawLive_vid_dict[this_trial_stim_num].get('Vid Count', 0) + 1
    this_trial_stim_vid = {}
    for tb, row in enumerate(this_trial_rawLive_vid_frames):
        tbucket_num = tb
        flattened_frame = row
        this_trial_stim_vid[tbucket_num] = flattened_frame
    for tbucket in this_trial_stim_vid.keys():
        if tbucket in daily_rawLive_vid_dict[this_trial_stim_num].keys():
            daily_rawLive_vid_dict[this_trial_stim_num][tbucket]['Trial Count'] = daily_rawLive_vid_dict[this_trial_stim_num][tbucket]['Trial Count'] + 1
            daily_rawLive_vid_dict[this_trial_stim_num][tbucket]['Summed Luminance'] = daily_rawLive_vid_dict[this_trial_stim_num][tbucket]['Summed Luminance'] + this_trial_stim_vid[tbucket]
        else: 
            daily_rawLive_vid_dict[this_trial_stim_num][tbucket] = {'Trial Count': 1, 'Summed Luminance': this_trial_stim_vid[tbucket]}

def calculate_meanPerDay_worldCam(day_worldCam_tbDict):
    # intialize time bucket dict for mean worldcam for each day
    meanPerDay_worldCam = {key:{'Vid Count':0} for key in stim_vids}
    for stim_num in day_worldCam_tbDict.keys():
        for key in day_worldCam_tbDict[stim_num].keys():
            if key == 'Vid Count':
                meanPerDay_worldCam[stim_num]['Vid Count'] = meanPerDay_worldCam[stim_num]['Vid Count'] + day_worldCam_tbDict[stim_num]['Vid Count']
            else: 
                meanFrame = day_worldCam_tbDict[stim_num][key]['Summed Frame'] / day_worldCam_tbDict[stim_num][key]['Trial Count']
                meanPerDay_worldCam[stim_num][key] = {'Mean Frame': meanFrame, 'Trial Count': day_worldCam_tbDict[stim_num][key]['Trial Count']}
    return meanPerDay_worldCam

def calculate_meanPerDay_rawLiveVid(day_rawLiveVid_tbDict):
    # intialize time bucket dict for mean worldcam for each day
    meanPerDay_rawLiveVid = {key:{'Vid Count':0} for key in stim_vids}
    for stim_num in day_rawLiveVid_tbDict.keys():
        for key in day_rawLiveVid_tbDict[stim_num].keys():
            if key == 'Vid Count':
                meanPerDay_rawLiveVid[stim_num]['Vid Count'] = meanPerDay_rawLiveVid[stim_num]['Vid Count'] + day_rawLiveVid_tbDict[stim_num]['Vid Count']
            else: 
                meanLuminance = day_rawLiveVid_tbDict[stim_num][key]['Summed Luminance'] / day_rawLiveVid_tbDict[stim_num][key]['Trial Count']
                meanPerDay_rawLiveVid[stim_num][key] = {'Mean Luminance': meanLuminance, 'Trial Count': day_rawLiveVid_tbDict[stim_num][key]['Trial Count']}
    return meanPerDay_rawLiveVid

def add_to_monthly_worldCam_dict(this_day_meanWorld_dict, this_day_month, monthly_world_vid_dict):
    for stim in this_day_meanWorld_dict.keys():
        if this_day_month in monthly_world_vid_dict[stim].keys():
            for timebucket in this_day_meanWorld_dict[stim].keys():
                if timebucket == 'Vid Count':
                    # collect vid count
                    monthly_world_vid_dict[stim][this_day_month]['Vid Count'] = monthly_world_vid_dict[stim][this_day_month]['Vid Count'] + this_day_meanWorld_dict[stim]['Vid Count']
                else:
                    monthly_world_vid_dict[stim][this_day_month][key] = monthly_world_vid_dict[stim][this_day_month][timebucket] + this_day_meanWorld_dict[stim][timebucket]
        else:
            for timebucket in this_day_meanWorld_dict[stim].keys():
                if timebucket == 'Vid Count':
                    monthly_world_vid_dict[stim][this_day_month] = {'Vid Count': this_day_meanWorld_dict[stim]['Vid Count']}
                else: 
                    monthly_world_vid_dict[stim][this_day_month][timebucket] = this_day_meanWorld_dict[stim][timebucket]

def add_to_monthly_rawLiveStim_dict(this_day_meanRawLive_dict, this_day_month, monthly_rawLive_vid_dict):
    for stim in this_day_meanRawLive_dict.keys():
        if this_day_month in monthly_rawLive_vid_dict[stim].keys():
            for timebucket in this_day_meanRawLive_dict[stim].keys():
                if timebucket == 'Vid Count':
                    monthly_rawLive_vid_dict[stim][this_day_month]['Vid Count'] = monthly_rawLive_vid_dict[stim][this_day_month]['Vid Count'] + this_day_meanRawLive_dict[stim]['Vid Count']
                else:
                    monthly_rawLive_vid_dict[stim][this_day_month][key] = monthly_rawLive_vid_dict[stim][this_day_month][timebucket] + this_day_meanRawLive_dict[stim][timebucket]
        else:
            for timebucket in this_day_meanRawLive_dict[stim].keys():
                if timebucket == 'Vid Count':
                    monthly_rawLive_vid_dict[stim][this_day_month] = {'Vid Count': this_day_meanRawLive_dict[stim]['Vid Count']}
                else: 
                    monthly_rawLive_vid_dict[stim][this_day_month][timebucket] = this_day_meanRawLive_dict[stim][timebucket]





def average_day_world_vids(day_world_vid_dict, day_date, avg_world_vid_dir, vid_height, vid_width):
    for stim in day_world_vid_dict.keys(): 
        print("Averaging world videos for stimuli {s}...".format(s=stim))
        avg_vid = []
        avg_vid.append([vid_height, vid_width])
        avg_vid.append([day_world_vid_dict[stim]['Vid Count']])
        for tbucket in day_world_vid_dict[stim].keys():
            if tbucket=='Vid Count':
                continue
            this_bucket = [tbucket]
            frame_count = day_world_vid_dict[stim][tbucket][0]
            summed_frame = day_world_vid_dict[stim][tbucket][1]
            avg_frame = summed_frame/frame_count
            avg_frame_list = avg_frame.tolist()
            for pixel in avg_frame_list:
                this_bucket.append(pixel)
            avg_vid.append(this_bucket)
        # save average world vid for each stimulus to csv
        avg_vid_csv_name = day_date + '_' + str(int(stim)) + '_Avg-World-Vid-tbuckets.csv'
        csv_file = os.path.join(avg_world_vid_dir, avg_vid_csv_name)
        print("Saving average world video of stimulus {s} for {d}".format(s=stim, d=day_date))
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerows(avg_vid)

def extract_daily_avg_world_vids(daily_avg_world_folder):
    stim_files = glob.glob(daily_avg_world_folder + os.sep + "*Avg-World-Vid-tbuckets.csv")
    world_vids_tbucketed = {}
    for stim_file in stim_files: 
        stim_name = stim_file.split(os.sep)[-1]
        stim_type = stim_name.split('_')[1]
        stim_number = np.float(stim_type)
        world_vids_tbucketed[stim_number] = {}
        extracted_rows = []
        with open(stim_file) as f:
            csvReader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            for row in csvReader:
                extracted_rows.append(row)
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
                world_vids_tbucketed[stim_number][tbucket_num] = flattened_frame
    return world_vids_tbucketed

def add_to_monthly_world_vids(analysis_folder_paths_for_month, list_of_stim_types):
    this_month_sum_world_vids = {key:{} for key in list_of_stim_types}
    this_month_vid_heights = []
    this_month_vid_widths = []
    for analysed_day in analysis_folder_paths_for_month:
        day_name = analysed_day.split(os.sep)[-1]
        print("Collecting world vid data from {day}".format(day=day_name))
        analysis_folder = os.path.join(analysed_day, "Analysis")
        world_folder = os.path.join(analysis_folder, "world")
        if not os.path.exists(world_folder):
            print("No average world frames exist for folder {name}!".format(name=day_name))
            continue
        this_day_avg_world_vids = extract_daily_avg_world_vids(world_folder)
        for stim_type in this_day_avg_world_vids.keys():
            this_month_sum_world_vids[stim_type] = {}
            this_stim_vid_height = this_day_avg_world_vids[stim_type]['Vid Dimensions'][0]
            this_month_vid_heights.append(this_stim_vid_height)
            this_stim_vid_width = this_day_avg_world_vids[stim_type]['Vid Dimensions'][1]
            this_month_vid_widths.append(this_stim_vid_width)
            this_stim_vid_count = this_day_avg_world_vids[stim_type]['Vid Count']
            this_month_sum_world_vids[stim_type]['Vid Count'] = this_month_sum_world_vids[stim_type].get('Vid Count', 0) + this_stim_vid_count
            for tbucket_num in this_day_avg_world_vids[stim_type].keys():
                if tbucket_num=='Vid Dimensions':
                    continue
                elif tbucket_num=='Vid Count':
                    continue
                elif tbucket_num in this_month_sum_world_vids[stim_type].keys():
                    this_month_sum_world_vids[stim_type][tbucket_num][0] = this_month_sum_world_vids[stim_type][tbucket_num][0] + 1
                    this_month_sum_world_vids[stim_type][tbucket_num][1] = this_month_sum_world_vids[stim_type][tbucket_num][1] + this_day_avg_world_vids[stim_type][tbucket_num]
                else:
                    this_month_sum_world_vids[stim_type][tbucket_num] = [1, this_day_avg_world_vids[stim_type][tbucket_num]]
    if not this_month_vid_heights:
            print("No world vids averaged for {date}".format(date=day_name))
    elif all(x == this_month_vid_heights[0] for x in this_month_vid_heights):
        if all(x == this_month_vid_widths[0] for x in this_month_vid_widths):
            this_month_vid_height = this_month_vid_heights[0]
            this_month_vid_width = this_month_vid_widths[0]
    else:
        print("Not all video heights and widths are equal!")
    return this_month_sum_world_vids, this_month_vid_height, this_month_vid_width

###################################
# SET CURRENT WORKING DIRECTORY
###################################
current_working_directory = os.getcwd()
###################################
# SCRIPT LOGGER
###################################
### log everything in a text file
class Logger(object):
    def __init__(self):
        # grab today's date
        now = datetime.datetime.now()
        todays_datetime = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')
        log_filename = "WorldVidExtraction_log_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
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
###################################
# DATA AND OUTPUT FILE LOCATIONS
###################################
# Synology drive
# on lab computer
#data_drive = r"\\Diskstation\SurprisingMinds"
#analysed_drive = r"C:\Users\Kampff_Lab\Dropbox\SurprisingMinds\analysis\dataPythonWorkflows"
# on laptop
data_drive = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\debuggingData"
analysed_drive = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\dataPythonWorkflows"
# collect input data subfolders
rawStimLum_data = os.path.join(analysed_drive, "rawStimLums")
analysed_folders = sorted(os.listdir(analysed_drive))
daily_csv_files = fnmatch.filter(analysed_folders, 'SurprisingMinds_*')
monthly_extracted_data = fnmatch.filter(analysed_folders, 'WorldVidAverage_*')

###################################
### ONLY RUN WHEN COMPLETELY RESTARTING WORLD VID PROCESSING (DELETES 'world' FOLDERS!!!)... 
###################################
for folder in daily_csv_files:
    subdirs = os.listdir(os.path.join(analysed_drive, folder, 'Analysis'))
    if 'world' in subdirs:
        shutil.rmtree(os.path.join(analysed_drive, folder, 'Analysis', 'world'))

###################################
# STIMULUS INFO
###################################
stim_vids = [24.0, 25.0, 26.0, 27.0, 28.0, 29.0]
stim_name_to_float = {"stimuli024": 24.0, "stimuli025": 25.0, "stimuli026": 26.0, "stimuli027": 27.0, "stimuli028": 28.0, "stimuli029": 29.0}
stim_float_to_name = {24.0: "stimuli024", 25.0: "stimuli025", 26.0: "stimuli026", 27.0: "stimuli027", 28.0: "stimuli028", 29.0: "stimuli029"}

###################################
# LOAD RAW VID STIM DATA
###################################
rawStimLum_files = glob.glob(rawStimLum_data + os.sep + '*.csv')
rawStimLum_dict = {}
for rSL_file in rawStimLum_files:
    stim_phase = os.path.basename(rSL_file).split('_')[0]
    stim_lums = np.genfromtxt(rSL_file, delimiter=',')
    thisPhase_lenFrames = len(stim_lums)
    rawStimLum_dict[stim_phase] = {'Number of Frames': thisPhase_lenFrames, 'Luminance per Frame': stim_lums}

###################################
# EXTRACT WORLD CAM VID TIMING AND LUMINANCE
# SAVE RAW LIVE VIDEOS FROM EACH TRIAL AS BINARY FILE
###################################
# get the subfolders, sort their names
data_folders = sorted(os.listdir(data_drive))
zipped_data = fnmatch.filter(data_folders, '*.zip')
# first day was debugging the exhibit
zipped_data = zipped_data[1:]
zipped_names = [item[:-4] for item in zipped_data]
# figure out which days have already been analysed
extracted_months = [item.split('_')[1] for item in monthly_extracted_data]
already_extracted_daily = []
for folder in daily_csv_files:
    subdirs = os.listdir(os.path.join(analysed_drive, folder, 'Analysis'))
    if 'world' in subdirs:
        already_extracted_daily.append(folder)

# DAYS THAT CANNOT BE UNZIPPED 
invalid_zipped = ['2017-12-28','2018-01-25']
# MONTHLY MEAN WORLD CAM
meanPerMonth_worldCam = {key:{} for key in stim_vids}
# MONTHLY MEAN RAW LIVE VID
meanPerMonth_rawLiveStim = {key:{} for key in stim_vids}
# BEGIN WORLD VID FRAME EXTRACTION/AVERAGING 
for item in zipped_data:
    this_day_date = item[:-4].split('_')[1]
    this_month = this_day_date[:-3]
    ########################################################################
    # check to see if this folder has already had world vid frames extracted
    ########################################################################
    if item[:-4] in already_extracted_daily:
        print("World vid frames from {name} has already been extracted".format(name=item))
        # check to see if this folder has already been averaged into a monthly world cam average
        item_year_month = this_day_date[:7]
        if item_year_month in extracted_months:
            print("World camera frames from {name} have already been consolidated into a monthly average".format(name=item_year_month))
            continue
        # if no monthly world cam average made yet for this month
        # check that the full month has been extracted by checking for a meanPerDay_worldCam binary file for each day


        this_month_extracted = fnmatch.filter(already_extracted_daily, 'SurprisingMinds_' + item_year_month + '*')
        this_month_data = fnmatch.filter(zipped_data, 'SurprisingMinds_' + item_year_month + '*')
        this_month_invalid = fnmatch.filter(invalid_zipped, item_year_month)
        if len(this_month_extracted) != len(this_month_data) + len(this_month_invalid):
            print("World vid frames for {month} not yet completed".format(month=item_year_month))
            continue
        # full month extracted?
        print('This month extraction completed: {month_list}'.format(month_list=this_month_extracted))
        
        # take world cam frames for each day and build a monthly average world cam sequence for each stim
        search_pattern = os.path.join(analysed_drive, 'SurprisingMinds_'+item_year_month+'-*')
        current_month_analysed = glob.glob(search_pattern)
        current_month_summed_world_vids, world_vid_height, world_vid_width = add_to_monthly_world_vids(current_month_analysed, stim_vids)
        average_monthly_world_vids(current_month_summed_world_vids, world_vid_height, world_vid_width, item_year_month, analysed_drive)
        
        # update list of already extracted months
        print("Updating list of extracted months...")
        analysed_folders = sorted(os.listdir(analysed_drive))
        monthly_extracted_data = fnmatch.filter(analysed_folders, 'WorldVidAverage_*')
        extracted_months = [item.split('_')[1] for item in monthly_extracted_data]
        # delete daily videos
        for daily_folder in current_month_analysed:
            current_date = daily_folder.split(os.sep)[-1].split('_')[1]
            analysis_folder = os.path.join(daily_folder, "Analysis")
            world_folder = os.path.join(analysis_folder, "world")
            print("Deleting daily world vid average files for {date}...".format(date=current_date))
            shutil.rmtree(world_folder)
            print("Delete successful!")
            print("Making empty 'world' folder for {date}...".format(date=current_date))
            os.makedirs(world_folder)
        print("Finished averaging world video frames for {month}!".format(month=item_year_month))
        continue
    #############################################################################
    # if world vid frames in this folder haven't already been extracted, EXTRACT!
    #############################################################################
    print("Extracting World Vid frames from folder {name}".format(name=item))
    # Build relative analysis paths, these folders should already exist
    analysis_folder = os.path.join(analysed_drive, item[:-4], "Analysis")
    alignment_folder = os.path.join(analysis_folder, "alignment")
    if not os.path.exists(analysis_folder):
        print("No Analysis folder exists for folder {name}!".format(name=item))
        continue
    # grab a folder 
    day_zipped = os.path.join(data_drive, item)
    # create Analysis subfolder for avg world vid data
    world_folder = os.path.join(analysis_folder, "world")
    # Create world_folder if it doesn't exist
    if not os.path.exists(world_folder):
        #print("Creating csv folder.")
        os.makedirs(world_folder)
    # create a temp folder in current working directory to store data (contents of unzipped folder)
    day_folder = os.path.join(current_working_directory, "world_temp")
    # at what time resolution to build raw live stim and world camera data?
    bucket_size = 4 #milliseconds
    # unzip current zipped folder into temp folder, this function checks whether the folder is unzippable
    # if it unzips, the function returns True; if it doesn't unzip, the function returns False
    if unpack_to_temp(day_zipped, day_folder):
        # List all trial folders
        trial_folders = list_sub_folders(day_folder)
        num_trials = len(trial_folders)
        # intialize time bucket dictionary for world vids
        this_day_worldCam_tbucket = {key:{'Vid Count':0} for key in stim_vids}
        this_day_world_vids_height = []
        this_day_world_vids_width = []
        # initialize time bucket dictionary for raw live stim vids
        this_day_rawLiveVid_tbucket = {key:{'Vid Count':0} for key in stim_vids}
        ###################################
        # extract world vid from each trial
        ###################################
        current_trial = 0
        for trial_folder in trial_folders:
            # add exception handling so that a weird day doesn't totally break everything 
            try:
                trial_name = trial_folder.split(os.sep)[-1]
                # check that the alignment frame for the day shows the correct start to the exhibit
                png_filename = trial_name + '.png'
                alignment_png_path = os.path.join(alignment_folder, png_filename)
                if os.path.exists(alignment_png_path):
                    alignment_img = mpimg.imread(alignment_png_path)
                    alignment_gray = cv2.cvtColor(alignment_img, cv2.COLOR_RGB2GRAY)
                    monitor_zoom = alignment_gray[60:-200, 110:-110]
                    monitor_score = np.sum(monitor_zoom)
                    # pick a pixel where it should be bright because people are centering their eyes in the cameras
                    if monitor_zoom[115,200]>=0.7:
                        ###################################
                        # Load CSVs and create timestamps
                        # ------------------------------
                        # Get world movie timestamp csv path
                        world_csv_path = glob.glob(trial_folder + '/*world.csv')[0]
                        # Get world video filepath
                        world_video_path = glob.glob(trial_folder + '/*world.avi')[0]
                        ####################################
                        # while debugging
                        #world_csv_path = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\debuggingData\SurprisingMinds_2017-10-14\2017-10-14_09-42-40\2017-10-14_09-42-40_stimuli024_world.csv"
                        #world_video_path = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\debuggingData\SurprisingMinds_2017-10-14\2017-10-14_09-42-40\2017-10-14_09-42-40_stimuli024_world.avi"
                        #world_folder = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\dataPythonWorkflows\SurprisingMinds_2017-10-14\Analysis\world"
                        ####################################
                        stimuli_name = world_csv_path.split("_")[-2]
                        stimuli_number = stim_name_to_float[stimuli_name]
                        # Load world CSV
                        world_timestamps = np.genfromtxt(world_csv_path, dtype=np.str, delimiter=' ') # row = timestamp, not frame
                        ### EXTRACT FRAMES FROM WORLD VIDS AND PUT INTO TIME BUCKETS ###
                        # create a "raw live stimulus video" array by combining framerate info from world cam with luminance values from raw vids
                        print("Extracting world vid frames and creating raw live stim vid for %s..." % os.path.basename(world_video_path))
                        # save raw live stim vid as binary files and return world cam frames as a sanity check
                        worldCam_vidWidth, worldCam_vidHeight, worldCam_supersampledFrames, rawLiveVid_supersampledFrames = supersampled_worldCam_rawLiveVid(world_video_path, world_timestamps, rawStimLum_dict, world_folder, bucket_size)
                        add_to_daily_worldCam_dict(worldCam_supersampledFrames, stimuli_number, this_day_worldCam_tbucket)
                        this_day_world_vids_width.append(worldCam_vidWidth)
                        this_day_world_vids_height.append(worldCam_vidHeight)
                        # ------------------------------
                        add_to_daily_rawLiveVid_dict(rawLiveVid_supersampledFrames, stimuli_number, this_day_rawLiveVid_tbucket)
                        # ------------------------------
                        # Report progress
                        cv2.destroyAllWindows()
                        print("Finished Trial: {trial}".format(trial=current_trial))
                        current_trial = current_trial + 1
                    else:
                        print("Bad trial! Stimulus did not display properly for trial {trial}".format(trial=current_trial))
                        current_trial = current_trial + 1
                else:
                    print("No alignment picture exists for trial {trial}".format(trial=current_trial))
                    current_trial = current_trial + 1
            except Exception: 
                cv2.destroyAllWindows()
                print("Trial {trial} failed!".format(trial=current_trial))
                current_trial = current_trial + 1
        ##################################################
        # check that all videos have same height and width
        ##################################################
        if not this_day_world_vids_height:
            print("No world vids averaged for {date}".format(date=this_day_date))
            # delete temporary file with unzipped data contents
            print("Deleting temp folder of unzipped data...")
            shutil.rmtree(day_folder)
            print("Delete successful!")
            continue
        if all(x == this_day_world_vids_height[0] for x in this_day_world_vids_height):
            if all(x == this_day_world_vids_width[0] for x in this_day_world_vids_width):
                unravel_height = this_day_world_vids_height[0]
                unravel_width = this_day_world_vids_width[0]
        ###########################################
        # average worldCam sanityCheck for each day
        ###########################################
        print('Calculating mean world camera videos for %s' % (this_day_date))
        thisDay_meanWorldCam = calculate_meanPerDay_worldCam(this_day_worldCam_tbucket)
        print('Adding daily mean world camera to monthly summed world camera video...')
        add_to_monthly_worldCam_dict(thisDay_meanWorldCam, this_day_date, meanPerMonth_worldCam)
        ###########################################
        # average rawLiveStim video for each day
        ###########################################
        print('Calculating mean raw live stim videos for %s' % (this_day_date))
        thisDay_meanRawLiveVid = calculate_meanPerDay_rawLiveVid(this_day_rawLiveVid_tbucket)
        print('Adding daily mean raw live stim vid to monthly summed raw live stim vids...')
        add_to_monthly_rawLiveStim_dict(thisDay_meanRawLiveVid, this_day_date, meanPerMonth_rawLiveStim)
        #################
        # report progress
        #################
        print("Finished extracting from {day}".format(day=day_zipped[:-4]))
        ###################################################
        # delete temporary file with unzipped data contents
        ###################################################
        print("Deleting temp folder of unzipped data...")
        shutil.rmtree(day_folder)
        print("Delete successful!")
    else:
        print("Could not unzip data folder for day {name}".format(name=this_day_date))
        invalid_zipped.append(this_day_date)
        print("Days that cannot be unzipped: {list}".format(list=invalid_zipped))
print("Completed world vid frame extraction on all data folders in this drive!")
# close logfile
sys.stdout.close()
#FIN