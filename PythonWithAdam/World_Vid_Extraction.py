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

def time_bucket_world_vid(video_path, video_timestamps, world_csv_path, bucket_size_ms):
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
    vid_length_tbuckets = len(time_chunks)
    frames = []
    for time in time_chunks:
        flattened_frame = stim_buckets[time]
        if not np.isnan(flattened_frame[0]):
            frames.append([time_chunks.index(time), flattened_frame])
    # release video capture
    world_vid.release()
    return vid_height, vid_width, vid_length_tbuckets, frames

def add_to_day_world_dict(this_trial_world_vid_frames, this_trial_stim_num, day_world_vid_dict):
    # keep track of how many videos are going into the average for this stim
    day_world_vid_dict[this_trial_stim_num]['Vid Count'] = day_world_vid_dict[this_trial_stim_num].get('Vid Count', 0) + 1
    this_trial_stim_vid = {}
    for row in this_trial_world_vid_frames:
        tbucket_num = row[0]
        flattened_frame = row[1]
        this_trial_stim_vid[tbucket_num] = flattened_frame
    for tbucket in this_trial_stim_vid.keys():
        if tbucket in day_world_vid_dict[this_trial_stim_num].keys():
            day_world_vid_dict[this_trial_stim_num][tbucket][0] = day_world_vid_dict[this_trial_stim_num][tbucket][0] + 1
            day_world_vid_dict[this_trial_stim_num][tbucket][1] = day_world_vid_dict[this_trial_stim_num][tbucket][1] + this_trial_stim_vid[tbucket]
        else: 
            day_world_vid_dict[this_trial_stim_num][tbucket] = [1, this_trial_stim_vid[tbucket]]

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
            if i==1:
                vid_count = int(extracted_rows[i][0])
                world_vids_tbucketed[stim_number]["Vid Count"] = vid_count
            else:
                tbucket_num = extracted_rows[i][0]
                flattened_frame = extracted_rows[i][1:]
                world_vids_tbucketed[stim_number][tbucket_num] = flattened_frame
    return world_vids_tbucketed

def add_to_monthly_world_vids(analysis_folder_paths_for_month, list_of_stim_types):
    this_month_sum_world_vids = {key:{} for key in list_of_stim_types}
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
            vid_height = this_day_avg_world_vids[stim_type]['Vid Dimensions'][0]
            vid_width = this_day_avg_world_vids[stim_type]['Vid Dimensions'][1]
            vid_count = this_day_avg_world_vids[stim_type]['Vid Count']
            this_month_sum_world_vids[stim_type]['Vid Count'] = this_month_sum_world_vids[stim_type].get('Vid Count', 0) + vid_count
            for tbucket_num in this_day_avg_world_vids[stim_type].keys():
                if tbucket_num=='Vid Dimensions':
                    continue
                if tbucket_num=='Vid Count':
                    continue
                if tbucket_num in this_month_sum_world_vids[stim_type].keys():
                    this_month_sum_world_vids[stim_type][tbucket_num][0] = this_month_sum_world_vids[stim_type][tbucket_num][0] + 1
                    this_month_sum_world_vids[stim_type][tbucket_num][1] = this_month_sum_world_vids[stim_type][tbucket_num][1] + this_day_avg_world_vids[stim_type][tbucket_num]
                else:
                    this_month_sum_world_vids[stim_type][tbucket_num] = [1, this_day_avg_world_vids[stim_type][tbucket_num]]
    return this_month_sum_world_vids, vid_height, vid_width

def average_monthly_world_vids(summed_monthly_world_vids_dict, vid_height, vid_width, month_name, analysed_data_drive):
    for stim in summed_monthly_world_vids_dict.keys(): 
        print("Averaging world videos for stimuli {s}...".format(s=stim))
        avg_vid = []
        avg_vid.append([vid_height, vid_width])
        avg_vid.append([[summed_monthly_world_vids_dict][stim]['Vid Count']])
        for tbucket in summed_monthly_world_vids_dict[stim].keys():
            this_bucket = [tbucket]
            frame_count = summed_monthly_world_vids_dict[stim][tbucket][0]
            summed_frame = summed_monthly_world_vids_dict[stim][tbucket][1]
            avg_frame = [i/frame_count for i in summed_frame]
            for pixel in avg_frame:
                this_bucket.append(pixel)
            avg_vid.append(this_bucket)
        # save average world vid for each stimulus to csv
        monthly_avg_vid_csv_name = month_name + '_Stimuli' + str(int(stim)) + '_Avg-World-Vid-tbuckets.csv'
        world_folder_name = 'WorldVidAverage_' + month_name
        world_folder_path = os.path.join(analysed_data_drive, world_folder_name)
        if not os.path.exists(world_folder_path):
            #print("Creating plots folder.")
            os.makedirs(world_folder_path)
        world_csv_filename = os.path.join(world_folder_path, monthly_avg_vid_csv_name)
        print("Saving average world video of stimulus {s} for {m}".format(s=stim, m=month_name))
        with open(world_csv_filename, 'w', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerows(avg_vid)

### -------------------------------------------- ###
### LET THE ANALYSIS BEGIN!! ###
### log everything in a text file
current_working_directory = os.getcwd()
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
### ------------------------------------------- ###
# list all folders in Synology drive
# on lab computer
data_drive = r"\\Diskstation\SurprisingMinds"
### FOR DEBUGGING ON LAPTOP ###
#data_drive = r'C:\Users\taunsquared\Desktop\SM_temp'
# get the subfolders, sort their names
data_folders = sorted(os.listdir(data_drive))
zipped_data = fnmatch.filter(data_folders, '*.zip')
# first day was debugging FOR LAB COMPUTER
zipped_data = zipped_data[1:]
zipped_names = [item[:-4] for item in zipped_data]
# figure out which days have already been analysed
# when working from local drive, lab computer
analysed_drive = r"C:\Users\KAMPFF-LAB-VIDEO\Dropbox\SurprisingMinds\analysis\pythonWithAdam-csv"
# when working from laptop
#analysed_drive = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\pythonWithAdam-csv"
analysed_folders = sorted(os.listdir(analysed_drive))
daily_csv_files = fnmatch.filter(analysed_folders, 'SurprisingMinds_*')
monthly_extracted_data = fnmatch.filter(analysed_folders, 'WorldVidAverage_*')
extracted_months = [item.split('_')[1] for item in monthly_extracted_data]
already_extracted_daily = []
for folder in daily_csv_files:
    subdirs = os.listdir(os.path.join(analysed_drive, folder, 'Analysis'))
    if 'world' in subdirs:
        already_extracted_daily.append(folder)
stim_vids = [24.0, 25.0, 26.0, 27.0, 28.0, 29.0]
stim_name_to_float = {"stimuli024": 24.0, "stimuli025": 25.0, "stimuli026": 26.0, "stimuli027": 27.0, "stimuli028": 28.0, "stimuli029": 29.0}
stim_float_to_name = {24.0: "stimuli024", 25.0: "stimuli025", 26.0: "stimuli026", 27.0: "stimuli027", 28.0: "stimuli028", 29.0: "stimuli029"}

# BEGIN WORLD VID FRAME EXTRACTION/AVERAGING #
for item in zipped_data:
    this_day_date = item[:-4].split('_')[1]
    # check to see if this folder has already had world vid frames extracted
    if item[:-4] in already_extracted_daily:
        print("World vid frames from {name} has already been extracted".format(name=item))
        # check to see if this folder has already been averaged into a monthly stim vid average
        item_year_month = this_day_date[:7]
        if item_year_month in extracted_months:
            print("World vid frames from {name} have already been consolidated into a monthly average".format(name=item_year_month))
            continue
        # if no monthly stim vid average made yet for this month
        # check that the full month has been extracted
        item_just_year = int(item_year_month.split('-')[0])
        item_just_month = int(item_year_month.split('-')[1])
        if item_just_month<=12:
            next_month = item_just_month + 1
            date_to_check = str(item_just_year) + '-' + str(next_month)
        else:
            next_year = item_just_year + 1
            next_month = '01'
            date_to_check = str(next_year) + '-' + str(next_month)
        next_month_analysed = fnmatch.filter(analysed_folders, 'SurprisingMinds_' + date_to_check + '*')
        if not next_month_analysed:
            print("World vid frames for {month} not yet completed".format(month=item_year_month))
            continue
        # full month extracted?
        # take avg stim vids for each day and build a monthly average vid for each stim
        search_pattern = os.path.join(analysed_drive, 'SurprisingMinds_'+item_year_month+'-*')
        current_month_analysed = glob.glob(search_pattern)
        current_month_summed_world_vids, world_vid_height, world_vid_width = add_to_monthly_world_vids(current_month_analysed, stim_vids)
        average_monthly_world_vids(current_month_summed_world_vids, world_vid_height, world_vid_width, item_year_month, analysed_drive)
        # delete daily videos
        for daily_folder in current_month_analysed:
            analysis_folder = os.path.join(daily_folder, item[:-4], "Analysis")
            world_folder = os.path.join(analysis_folder, "world")
            print("Deleting daily world vid average files...")
            shutil.rmtree(world_folder)
            print("Delete successful!")
            print("Making empty 'world' folder...")
            os.makedirs(world_folder)
            print("Finished!")
        continue
    
    # if world vid frames this folder haven't already been extracted, EXTRACT!
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
        # intialize time bucket dictionary for world vids
        this_day_world_vids_tbucket = {key:{} for key in stim_vids}
        this_day_world_vids_height = []
        this_day_world_vids_width = []
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
                    # pick a pixel where it should be bright because people are centering their eyes in the cameras
                    if monitor_zoom[115,200]>=0.7:
                        # Load CSVs and create timestamps
                        # ------------------------------
                        # Get world movie timestamp csv path
                        world_csv_path = glob.glob(trial_folder + '/*world.csv')[0]
                        stimuli_name = world_csv_path.split("_")[-2]
                        stimuli_number = stim_name_to_float[stimuli_name]
                        # at what time resolution to build eye and world camera data?
                        bucket_size = 4 #milliseconds

                        # Load world CSV
                        world_timestamps = np.genfromtxt(world_csv_path, dtype=np.str, delimiter=' ')
                        # Get world video filepath
                        world_video_path = glob.glob(trial_folder + '/*world.avi')[0]
                        ### EXTRACT FRAMES FROM WORLD VIDS AND PUT INTO TIME BUCKETS ###
                        print("Extracting world vid frames...")
                        # save this to an array and accumulate over trials
                        world_vid_height, world_vid_width, world_vid_length_tbuckets, world_vid_frames = time_bucket_world_vid(world_video_path, world_timestamps, world_folder, bucket_size)
                        this_day_world_vids_height.append(world_vid_height)
                        this_day_world_vids_width.append(world_vid_width)
                        add_to_day_world_dict(world_vid_frames, stimuli_number, this_day_world_vids_tbucket)
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

        # check that all videos have same height and width
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
                vid_count = len(this_day_world_vids_height)
        # average and save world videos for each stimulus type
        average_day_world_vids(this_day_world_vids_tbucket, this_day_date, world_folder, unravel_height, unravel_width)
        # report progress
        print("Finished extracting from {day}".format(day=day_zipped[:-4]))
        # delete temporary file with unzipped data contents
        print("Deleting temp folder of unzipped data...")
        shutil.rmtree(day_folder)
        print("Delete successful!")

#FIN
print("Completed world vid frame extraction on all data folders in this drive!")
# close logfile
sys.stdout.close()