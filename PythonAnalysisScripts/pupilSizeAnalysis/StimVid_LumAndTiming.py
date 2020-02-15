### ------------------------------------------------------------------------- ###
### Create binary files of raw stim vid luminance values fitted to world cam stim vid presentation timings
### use world camera vids for timing, use raw vid luminance values extracted via bonsai
### also save world cam luminance as sanity check/ground truth
### also count how many times each language was chosen
### output as data files categorized by calibration, octopus, and unique sequences.
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
    # save rawLiveVid and worldCam sanityCheck
    rawLiveVid_output = output_folder + os.sep + video_name[:-10] + '_rawLiveVid_%dmsTBs.npy' % (bucket_size_ms)
    worldCam_output = output_folder + os.sep + video_name[:-10] + '_worldCamSanityCheck_%dmsTBs_%dBy%d.npy' % (bucket_size_ms, vid_width, vid_height)
    np.save(rawLiveVid_output, supersampled_rawLiveVid_array)
    np.save(worldCam_output, supersampled_worldCam_array)

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
# for folder in daily_csv_files:
#     subdirs = os.listdir(os.path.join(analysed_drive, folder, 'Analysis'))
#     if 'world' in subdirs:
#         os.rmdir(os.path.join(analysed_drive, folder, 'Analysis', 'world'))
#     if 'npy' in subdirs:
#         os.rmdir(os.path.join(analysed_drive, folder, 'Analysis', 'npy'))

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
# BEGIN WORLD VID FRAME EXTRACTION/AVERAGING 
for item in zipped_data:
    this_day_date = item[:-4].split('_')[1]
    # check to see if this folder has already had world vid frames extracted
    if item[:-4] in already_extracted_daily:
        print("World vid frames from {name} has already been extracted".format(name=item))
        continue
    # if world vid frames in this folder haven't already been extracted, EXTRACT!
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
    # unzip current zipped folder into temp folder, this function checks whether the folder is unzippable
    # if it unzips, the function returns True; if it doesn't unzip, the function returns False
    if unpack_to_temp(day_zipped, day_folder):
        # List all trial folders
        trial_folders = list_sub_folders(day_folder)
        num_trials = len(trial_folders)
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
                        # calculate language choice - STILL NEED TO DEBUG
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
                        # at what time resolution to build eye and world camera data?
                        bucket_size = 4 #milliseconds
                        # Load world CSV
                        world_timestamps = np.genfromtxt(world_csv_path, dtype=np.str, delimiter=' ') # row = timestamp, not frame
                        ### EXTRACT FRAMES FROM WORLD VIDS AND PUT INTO TIME BUCKETS ###
                        # create a "raw live stimulus video" array by combining framerate info from world cam with luminance values from raw vids
                        print("Extracting world vid frames and creating raw live stim vid for %s..." % os.path.basename(world_video_path))
                        # save raw live stim vid and save world cam frames as a sanity check
                        supersampled_worldCam_rawLiveVid(world_video_path, world_timestamps, rawStimLum_dict, world_folder, bucket_size)
                        # ------------------------------
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
        # report progress
        print("Finished extracting from {day}".format(day=day_zipped[:-4]))
        # delete temporary file with unzipped data contents
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