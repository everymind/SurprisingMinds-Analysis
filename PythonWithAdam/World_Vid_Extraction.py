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

### FUNCTIONS ###
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
    # -5 remains in a time bucket, this means no 'near-enough timestamp' frame was found in video

    return bucket_list

def find_nearest_timestamp_key(timestamp_to_check, dict_of_timestamps, time_window):
    for key in dict_of_timestamps.keys():
        if key <= timestamp_to_check <= (key + time_window):
            return key

def save_average_clip_images(which_eye, no_of_seconds, save_folder_path, images):
    # Save images from trial clip to folder
    #print("Saving averaged frames from {eye}...".format(eye=which_eye))
    for f in range(no_of_seconds):

        # Create file name with padded zeros
        padded_filename = which_eye + str(f).zfill(4) + ".png"

        # Create image file path from save folder
        image_file_path = os.path.join(save_folder_path, padded_filename)

        # Extract gray frame from clip
        gray = np.uint8(images[:,:,f] * 255)

        # Write to image file
        ret = cv2.imwrite(image_file_path, gray)

def time_bucket_world_vid(video_path, video_timestamps, npy_path, bucket_size_ms):
    ### row = timestamp, not frame #
    # Open world video
    world_vid = cv2.VideoCapture(video_path)
    vid_width = int(world_vid.get(3))
    vid_height = int(world_vid.get(4))
    # Get video file details
    video_name = video_path.split(os.sep)[-1]
    video_date = video_name.split('_')[0]
    video_time = video_name.split('_')[1]
    video_stim_number = video_name.split('_')[2]
    # each time bucket = 4ms (world cameras ran at approx 30fps, aka 33.333 ms per frame)
    first_timestamp = video_timestamps[0]
    last_timestamp = video_timestamps[-1]
    initialize_pattern = np.empty((vid_height,vid_width))
    initialize_pattern[:] = np.nan
    stim_buckets = make_time_buckets(first_timestamp, bucket_size_ms, last_timestamp, initialize_pattern)
    # Loop through 4ms time buckets of world video to find nearest frame and save 2-d matrix of pixel values in that frame
    for timestamp in video_timestamps:
        # find the time bucket into which this frame falls
        timestamp = timestamp.split('+')[0][:-3]
        timestamp_dt = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f")
        bucket_window = datetime.timedelta(milliseconds=bucket_size_ms)
        current_key = find_nearest_timestamp_key(timestamp_dt, stim_buckets, bucket_window)
        # Read frame at current position
        # should this be at current key??
        ret, frame = world_vid.read()
        # Make sure the frame exists!
        if frame is not None:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # append to dictionary stim_buckets
            stim_buckets[current_key] = gray
    time_chunks = []
    for key in stim_buckets.keys():
        time_chunks.append(key)
    time_chunks = sorted(time_chunks)
    frames = []
    for time in time_chunks:
        frame_array = stim_buckets[time]
        frames.append(frame_array)
    # save world vid frame data to numpy binary file
    padded_filename = video_date + "_" + video_time + "_" + video_stim_number + "_world-tbuckets.npy"
    npy_file = os.path.join(npy_path, padded_filename)
    np.save(npy_file, frames, allow_pickle=False, fix_imports=False)
    # release video capture
    world_vid.release()

### -------------------------------------------- ###
### LET THE ANALYSIS BEGIN!! ###
### log everything in a text file
current_working_directory = os.getcwd()
class Logger(object):
    def __init__(self):
        # grab today's date
        now = datetime.datetime.now()
        todays_datetime = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')
        log_filename = "pupil-plotting_log_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
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
### ------------------------------------------- ###
# list all folders in Synology drive
data_drive = r"\\Diskstation\SurprisingMinds"
# get the subfolders, sort their names
data_folders = sorted(os.listdir(data_drive))
zipped_data = fnmatch.filter(data_folders, '*.zip')
zipped_names = [item[:-4] for item in zipped_data]
# figure out which days have already had world vid frames extracted
# when working from local drive, lab computer
analysed_drive = r"C:\Users\KAMPFF-LAB-VIDEO\Dropbox\SurprisingMinds\analysis\pythonWithAdam-csv"
# when working from laptop
#analysed_drive = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\pythonWithAdam-csv"
analysed_folders = sorted(os.listdir(analysed_drive))
already_extracted = []
for folder in analysed_folders:
    subdirs = os.listdir(os.path.join(analysed_drive, folder, 'Analysis'))
    if 'npy' in subdirs:
        already_extracted.append(folder)
# unzip each folder, do the extraction
for item in zipped_data:

    # check to see if this folder has already been analyzed
    if item[:-4] in already_extracted:
        print("World video frames for {name} have already been extracted".format(name=item))
        continue
    
    # if this day hasn't had the world vid frames extracted, full speed ahead!
    print("Working on folder {name}".format(name=item))

    # grab a folder 
    day_zipped = os.path.join(data_drive, item)

    # Build relative world vid frames extraction path in a folder with same name as zip folder
    analysis_folder = os.path.join(analysed_drive, item[:-4], "Analysis")
    # extraction subfolder
    npy_folder = os.path.join(analysis_folder, "npy")

    # Create analysis and extraction folders if they do not exist
    if not os.path.exists(analysis_folder):
        #print("Creating analysis folder.")
        os.makedirs(analysis_folder)
    if not os.path.exists(npy_folder):
        #print("Creating csv folder.")
        os.makedirs(npy_folder)

    # create a temp folder in current working directory to store data (contents of unzipped folder)
    day_folder = os.path.join(current_working_directory, "temp")

    # unzip current zipped folder into temp folder, this function checks whether the folder is unzippable
    # if it unzips, the function returns True; if it doesn't unzip, the function returns False
    if unpack_to_temp(day_zipped, day_folder):

        # List all trial folders
        trial_folders = list_sub_folders(day_folder)
        num_trials = len(trial_folders)

        # Load all right eye movies and average
        current_trial = 0

        for trial_folder in trial_folders:
            # add exception handling so that a weird day doesn't totally break everything 
            try:
                trial_name = trial_folder.split(os.sep)[-1]
                # Load CSVs and create timestamps
                # ------------------------------
                #print("Loading csv files for {trial}...".format(trial=trial_name))
                # Get world movie timestamp csv path
                world_csv_path = glob.glob(trial_folder + '/*world.csv')[0]
                stimuli_number = world_csv_path.split("_")[-2]
                # at what time resolution to build eye and world camera data?
                bucket_size = 4 #milliseconds
                # Load world CSV
                world_timestamps = np.genfromtxt(world_csv_path, dtype=np.str, delimiter=' ')
                # Get world video filepath
                world_video_path = glob.glob(trial_folder + '/*world.avi')[0]
                # Open world video
                world_video = cv2.VideoCapture(world_video_path)
                ### EXTRACT FRAMES FROM WORLD VIDS AND PUT INTO TIME BUCKETS ###
                print("Extracting world vid frames...")
                time_bucket_world_vid(world_video_path, world_timestamps, npy_folder, bucket_size)
                
                # Report progress
                cv2.destroyAllWindows()
                print("Finished extracting world vid frames from trial: {trial}".format(trial=current_trial))
                current_trial = current_trial + 1
            except Exception: 
                cv2.destroyAllWindows()
                print("Trial {trial} failed!".format(trial=current_trial))
                current_trial = current_trial + 1

        # report progress
        world_video.release()
        cv2.destroyAllWindows()
        print("Finished {day}".format(day=day_zipped[:-4]))

        # delete temporary file with unzipped data contents
        print("Deleting temp folder of unzipped data...")
        shutil.rmtree(day_folder)
        print("Delete successful!")

#FIN
print("Completed extraction on all data folders in this drive!")
# close logfile
sys.stdout.close()