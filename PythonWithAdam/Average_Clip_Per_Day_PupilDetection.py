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
import csv

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

def find_darkest_circle(list_of_circles, source_image):
    #print("Finding darkest circle in {list}...".format(list=list_of_circles))
    # starting parameters
    darkest_intensity = 255
    darkest_index = 0
    # check that source_image is a grayscaled image
    if len(source_image.shape) > 2: 
        print("{Image} is not grayscale!".format(Image=source_image))
        exit()
    for i in range(len(list_of_circles)):
        # create a mask image that is the same size as source_image
        mask = np.zeros(source_image.shape, source_image.dtype)
        # get center coordinates and radius of circle from list_of_circle
        center = (list_of_circles[i][0], list_of_circles[i][1])
        radius = list_of_circles[i][2]
        #print("Center: {x},{y}".format(x=center[0], y=center[1]))
        # draw mask circle at coordinates and w/radius of circle from list_of_circles
        mask_circle = cv2.circle(mask, center, radius, 255, -1)
        ## for debugging
        # this_circle = cv2.circle(source_image, center, radius, (i*100, 0, 100), 2)
        # plt.imshow(source_image)
        # plt.show(block=False)
        # plt.pause(2)
        # plt.clf()
        # plt.cla()
        # plt.close()
        # get coordinates of mask circle pixels
        where = np.where(mask==255)
        # find those same coordinates in source_image
        intensity_inside_circle_on_source_image = source_image[where[0], where[1]]
        # take average of those pixels in source_image
        average_intensity = np.average(intensity_inside_circle_on_source_image)
        #print("Average intensity of circle {number}: {intensity}".format(number=i, intensity=average_intensity))
        # check this circle's intensity against darkest circle found so far
        if (average_intensity < darkest_intensity):
            darkest_intensity = average_intensity
            darkest_index = i
    #print("Darkest circle: {number}, intensity {intensity}".format(number=darkest_index, intensity=darkest_intensity))
    return list_of_circles[darkest_index]

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

def find_pupil(which_eye, which_stimuli, trial_number, video_path, video_timestamps, align_frame, csv_path, bucket_size_ms):
    ### row = timestamp, not frame #
    # Open eye video and world video
    video = cv2.VideoCapture(video_path)
    # Jump to specific frame (position) for alignment purposes 
    ret = video.set(cv2.CAP_PROP_POS_FRAMES, align_frame)
    # Open display window for debugging
    video_name = video_path.split(os.sep)[-1]
    debug_name = "Eye"+"_"+video_name
    cv2.namedWindow(debug_name)
    # each time bucket = 4ms (eye cameras ran at 60fps, aka 16.6666 ms per frame)
    # octobpus clip to thank you screen is 16.2 seconds
    first_timestamp = video_timestamps[align_frame]
    last_timestamp = video_timestamps[-1]
    initialize_pattern = [-5,-5,-5,-5,-5,-5]
    pupil_buckets = make_time_buckets(first_timestamp, bucket_size_ms, last_timestamp, initialize_pattern)
    
    # Loop through 4ms time buckets of eye video to find nearest frame and save pupil xy positon and area
    timestamps_to_check = video_timestamps[align_frame:]
    for timestamp in timestamps_to_check:
        # find the time bucket into which this frame falls
        timestamp = timestamp.split('+')[0][:-3]
        timestamp_dt = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f")
        bucket_window = datetime.timedelta(milliseconds=bucket_size_ms)
        current_key = find_nearest_timestamp_key(timestamp_dt, pupil_buckets, bucket_window)
        # Read frame at current position
        ret, frame = video.read()
        mask = np.copy(frame)
        # Make sure the frame exists!
        if frame is not None:
            # Magically find pupil...
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # Median blur
            blurred = cv2.medianBlur(gray, 25)
            # Hough circle detection
            rows = blurred.shape[0]
            ## WTF DOES HOUGHCIRCLES DO??
            ## sometimes the image seems really clean and easy to find the pupil and yet it still fails
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.0, rows / 8,
                                    param1=75, param2=25,
                                    minRadius=10, maxRadius=150)
            # If there are no circles, then what??
            if circles is not None:
                #print("Circles found: {circles}".format(circles=circles))
                # check that we are taking the darkest circle
                darkest_circle = find_darkest_circle(circles[0], blurred)
                #print("Darkest circle: {circle}".format(circle=darkest_circle))
                # Using the best circle...crop around center
                # Threshold
                # Fit an ellipse
                # Crop
                eye_circle = np.uint16(np.around(darkest_circle))
                left = eye_circle[0] - 64
                top = eye_circle[1] - 64
                crop_size = 128
                # Check boundarys of image
                if( (left >= 0) and (top >= 0) and ((left + crop_size) < 800) and ((top + crop_size) < 600) ):
                    cropped = gray[top:(top + crop_size), left:(left+crop_size)]
                    # Compute average and stdev of all pixel luminances along border
                    ## this currently averages the rightmost and leftmost edges of the cropped window, because we assume that these pixels are not the pupil
                    avg = (np.mean(cropped[:, 0]) + np.mean(cropped[:, -1])) / 2
                    std = (np.std(cropped[:, 0]) + np.std(cropped[:, -1])) / 2
                    ## Find shape of pupil
                    # Threshold
                    ## try removing otsu
                    ## try using 2 standard devs away from average instead of 3
                    thresholded = np.uint8(cv2.threshold(cropped, avg-(std*3), 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1])
                    # Find contours
                    contours, heirarchy = cv2.findContours(thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    # if more than one contour
                    if len(contours) > 0:
                        # Get largest contour
                        largest_contour = max(contours, key=cv2.contourArea)
                        # sanity check size of largest contour
                        ## SHOULD MAKE SURE THAT LARGEST CONTOUR ISN'T BIGGER THAN CROPPED
                        #####
                        # make sure contour is large enough to fit an ellipse to it
                        if(len(largest_contour) > 5):
                            # Fit ellipse to largest contour
                            ellipse = cv2.fitEllipse(largest_contour)
                            # Shift ellipse back to full frame coordinates
                            shifted_center = (np.int(ellipse[0][0]) + left, np.int(ellipse[0][1]) + top)
                            # Draw circles
                            circles = np.uint16(np.around(circles))
                            for i in circles[0, :]:
                                center = (i[0], i[1])
                                # circle center
                                cv2.circle(frame, center, 5, (0, 100, 100), 1)
                                # circle outline
                                radius = i[2]
                                cv2.circle(frame, center, radius, (255, 0, 255), 1)
                            # Draw ellipse around largest contour
                            axes = (np.int(ellipse[1][0]/2),np.int(ellipse[1][1]/2)) 
                            angle = np.int(ellipse[2])
                            frame = cv2.ellipse(frame, shifted_center, axes, angle, 0, 360, (0, 255, 0), 3, cv2.LINE_AA, 0)
                            # Draw debugging circle around darkest circle
                            axes = (darkest_circle[2], darkest_circle[2]) 
                            angle = 0
                            frame = cv2.ellipse(frame, (darkest_circle[0], darkest_circle[1]), axes, angle, 0, 360, (0, 0, 255), 2, cv2.LINE_AA, 0)
                            # Save Data
                            darkest_circle_area = np.pi*(darkest_circle[2])**2
                            #print("Pupil Size predicted by ellipses: {area}".format(area=cv2.contourArea(largest_contour)))
                            #print("Pupil size predicted by circles: {area1}".format(area1=darkest_circle_area))
                            # save data from both findContours and find_darkest_circle
                            pupil_buckets[current_key][0] = shifted_center[0]
                            pupil_buckets[current_key][1] = shifted_center[1]
                            pupil_buckets[current_key][2] = cv2.contourArea(largest_contour)
                            pupil_buckets[current_key][3] = darkest_circle[0]
                            pupil_buckets[current_key][4] = darkest_circle[1]
                            pupil_buckets[current_key][5] = (darkest_circle[2]**2) * math.pi
                            # Fill debug displays and show
                            cv2.imshow(debug_name, frame)
                            ret = cv2.waitKey(1)
                        else:
                            #print("Pupil Size: n/a (too small)")
                            pupil_buckets[current_key][2] = -1
                            pupil_buckets[current_key][5] = -1
                    else:
                        #print("Pupil Size: n/a (pupil off screen)")
                        pupil_buckets[current_key][2] = -2
                        pupil_buckets[current_key][5] = -2
                else:
                    #print("Pupil Size: n/a (no contour)")
                    pupil_buckets[current_key][2] = -3
                    pupil_buckets[current_key][5] = -3
            else:
                #print("Pupil Size: n/a (no circles)")
                pupil_buckets[current_key][2] = -4
                pupil_buckets[current_key][5] = -4
            ## STILL DOING THIS?????
            # Add current frame to average clip at correct slot
            #day_avg_clip[:,:,f] = day_avg_clip[:,:,f] + gray
    # Save pupil size data
    # HOW TO SAVE A DICTIONARY AS A CSV????
    time_chunks = []
    for key in pupil_buckets.keys():
        time_chunks.append(key)
    time_chunks = sorted(time_chunks)
    pupils = []
    for time in time_chunks:
        pupil = pupil_buckets[time]
        pupils.append(pupil)
    #print("Saving csv of positions and areas for {eye} eye...".format(eye=which_eye))
    padded_filename = which_eye + "_" + which_stimuli + "_" + str(trial_number).zfill(4) + ".csv"
    csv_file = os.path.join(csv_path, padded_filename)
    np.savetxt(csv_file, pupils, fmt='%.2f', delimiter=',')
    # release video capture
    video.release()
    cv2.destroyAllWindows()

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

def time_bucket_world_vid(video_path, video_timestamps, csv_path, bucket_size_ms):
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
    initialize_pattern = np.empty((vid_height*vid_width,))
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
            # flatten the frame into a list
            flattened_gray = gray.ravel()
            flattened_gray = flattened_gray.astype(None)
            # append to dictionary stim_buckets
            stim_buckets[current_key] = flattened_gray
    time_chunks = []
    for key in stim_buckets.keys():
        time_chunks.append(key)
    time_chunks = sorted(time_chunks)
    frames = []
    # append original video dimensions, for reshaping flattened frame arrays later
    frames.append([vid_height, vid_width])
    for time in time_chunks:
        flattened_frame = stim_buckets[time]
        if not np.isnan(flattened_frame[0]):
            frames.append([time_chunks.index(time), flattened_frame])
    # save world vid frame data to numpy binary file
    padded_filename = video_date + "_" + video_time + "_" + video_stim_number + "_world-tbuckets.csv"
    csv_file = os.path.join(csv_path, padded_filename)
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(frames)
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
# figure out which days have already been analysed
# when working from local drive, lab computer
analysed_drive = r"C:\Users\KAMPFF-LAB-VIDEO\Dropbox\SurprisingMinds\analysis\pythonWithAdam-csv"
# when working from laptop
#analysed_drive = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\pythonWithAdam-csv"
analysed_folders = sorted(os.listdir(analysed_drive))
already_analysed = [item for item in zipped_names if item in analysed_folders]
# unzip each folder, do the analysis
for item in zipped_data:

    # check to see if this folder has already been analyzed
    if item[:-4] in already_analysed:
        print("Folder {name} has already been analysed".format(name=item))
        continue
    
    # if this folder hasn't already been analysed, full speed ahead!
    print("Working on folder {name}".format(name=item))

    # grab a folder 
    day_zipped = os.path.join(data_drive, item)

    # Build relative analysis paths in a folder with same name as zip folder

    # when keeping analysis csv files in data_drive folder
    # analysis_folder = os.path.join(day_zipped[:-4], "Analysis")
    # when immediately placing analysis csv files in analysed drive
    analysis_folder = os.path.join(analysed_drive, item[:-4], "Analysis")
    # when debugging
    #analysis_folder = os.path.join(current_working_directory, item[:-4], "Analysis")

    # Analysis subfolders
    csv_folder = os.path.join(analysis_folder, "csv")
    alignment_folder = os.path.join(analysis_folder, "alignment")

    # Create analysis folder (and sub-folders) if it (they) does (do) not exist
    if not os.path.exists(analysis_folder):
        #print("Creating analysis folder.")
        os.makedirs(analysis_folder)
    if not os.path.exists(csv_folder):
        #print("Creating csv folder.")
        os.makedirs(csv_folder)
    if not os.path.exists(alignment_folder):
        #print("Creating alignment folder.")
        os.makedirs(alignment_folder)

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

                # Get eye timestamp csv paths
                right_eye_csv_path = glob.glob(trial_folder + '/*righteye.csv')[0]
                left_eye_csv_path = glob.glob(trial_folder + '/*lefteye.csv')[0]

                # Load eye CSVs
                right_eye_timestamps = np.genfromtxt(right_eye_csv_path, dtype=np.str, delimiter=' ')
                left_eye_timestamps = np.genfromtxt(left_eye_csv_path, dtype=np.str, delimiter=' ')
                # Get world video filepath
                world_video_path = glob.glob(trial_folder + '/*world.avi')[0]
                # Open world video
                world_video = cv2.VideoCapture(world_video_path)
                ### NOW WE ARE FINDING PUPILS FOR THE WHOLE STIMULI SEQUENCE ###
                # Show the frame to check where we are starting pupil finding (ground truth)
                fig_name = trial_name + ".png"
                fig_path = os.path.join(alignment_folder, fig_name)
                ret, frame = world_video.read()
                plt.imshow(frame)
                plt.savefig(fig_path)
                plt.show(block=False)
                plt.pause(1)
                plt.close()
                # ------------------------------
                world_video.release()
                ### EXTRACT FRAMES FROM WORLD VIDS AND PUT INTO TIME BUCKETS ###
                print("Extracting world vid frames...")
                time_bucket_world_vid(world_video_path, world_timestamps, csv_folder, bucket_size)
                # ------------------------------
                # ------------------------------
                # Now start pupil detection                
                # ------------------------------
                # Get right eye video filepath
                right_video_path = glob.glob(trial_folder + '/*righteye.avi')[0]
                # Get left eye video filepath
                left_video_path = glob.glob(trial_folder + '/*lefteye.avi')[0]
                
                # Find right eye pupils and save pupil data
                print("Finding right eye pupils...")
                # find_pupil(which_eye, which_stimuli, trial_number, video_path, video_timestamps, align_frame, csv_path)
                #find_pupil("right", stimuli_number, current_trial, right_video_path, right_eye_timestamps, right_eye_octopus, csv_folder)
                # NOW WE ARE FINDING PUPILS FOR THE WHOLE STIMULI SEQUENCE
                find_pupil("right", stimuli_number, current_trial, right_video_path, right_eye_timestamps, 0, csv_folder, bucket_size)
                # Find left eye pupils and save pupil data
                print("Finding left eye pupils...")
                #find_pupil("left", stimuli_number, current_trial, left_video_path, left_eye_timestamps, left_eye_octopus, csv_folder)
                find_pupil("left", stimuli_number, current_trial, left_video_path, left_eye_timestamps, 0, csv_folder, bucket_size)
                
                # Report progress
                cv2.destroyAllWindows()
                print("Finished Trial: {trial}".format(trial=current_trial))
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
print("Completed analysis on all data folders in this drive!")
# close logfile
sys.stdout.close()