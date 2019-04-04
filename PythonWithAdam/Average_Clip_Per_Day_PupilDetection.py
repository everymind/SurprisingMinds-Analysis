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

### FUNCTIONS ###

def unpack_to_temp(path_to_zipped, path_to_temp):
    try:
        # copy zip file to current working directory
        print("Copying {folder} to current working directory...".format(folder=path_to_zipped))
        current_working_directory = os.getcwd()
        copied_zipped = shutil.copy2(path_to_zipped, current_working_directory)
        path_to_copied_zipped = os.path.join(current_working_directory, copied_zipped.split(sep=os.sep)[-1])
        # unzip the folder
        print("Unzipping files in {folder}...".format(folder=path_to_copied_zipped))
        day_unzipped = zipfile.ZipFile(path_to_copied_zipped, mode="r")
        # extract files into temp folder
        day_unzipped.extractall(path_to_temp)
        # close the unzipped file
        day_unzipped.close()
        print("Finished unzipping {folder}!".format(folder=path_to_copied_zipped))
        # destroy copied zipped file
        print("Deleting {file}...".format(file=path_to_copied_zipped))
        os.remove(path_to_copied_zipped)
        print("Deleted {file}!".format(file=path_to_copied_zipped))
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
    # Find the frame that best matches the target time (seconds) from the start frame
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
    print("Finding darkest circle in {list}...".format(list=list_of_circles))
    # starting parameters
    darkest_intensity = 255
    darkest_index = 0
    # check that source_image is a grayscaled image
    if len(source_image.shape) > 2: 
        print("{Image} is not grayscale!".format(Image=source_image))
        exit()
    for i in range(len(list_of_circles)):
        print("Creating mask image...")
        # create a mask image that is the same size as source_image
        mask = np.zeros(source_image.shape, source_image.dtype)
        # get center coordinates and radius of circle from list_of_circle
        center = (list_of_circles[i][0], list_of_circles[i][1])
        radius = list_of_circles[i][2]
        print("Center: {x},{y}".format(x=center[0], y=center[1]))
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
        print("Average intensity of circle {number}: {intensity}".format(number=i, intensity=average_intensity))
        # check this circle's intensity against darkest circle found so far
        if (average_intensity < darkest_intensity):
            darkest_intensity = average_intensity
            darkest_index = i
    print("Darkest circle: {number}, intensity {intensity}".format(number=darkest_index, intensity=darkest_intensity))
    return list_of_circles[darkest_index]

def find_pupil(which_eye, trial_number, video_path, align_frame, no_of_frames, day_avg_clip, csv_path):
    # Open eye video and world video
    video = cv2.VideoCapture(video_path)
    # Jump to specific frame (position) for alignment purposes 
    ret = video.set(cv2.CAP_PROP_POS_FRAMES, align_frame)
    # Open display window for debugging
    cv2.namedWindow("Eye")
    # Create empty data array
    pupil = np.zeros((no_of_frames, 3))
    # Loop through frames of eye video to find and save pupil xy positon and area
    for f in range(0, no_of_frames, 1):
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
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.0, rows / 8,
                                    param1=75, param2=25,
                                    minRadius=10, maxRadius=150)
            # If there are no circles, then what??
            if circles is not None:
                print("Circles found: {circles}".format(circles=circles))
                # check that we are taking the darkest circle
                darkest_circle = find_darkest_circle(circles[0], blurred)
                print("Darkest circle: {circle}".format(circle=darkest_circle))
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
                    avg = (np.mean(cropped[:, 0]) + np.mean(cropped[:, -1])) / 2
                    std = (np.std(cropped[:, 0]) + np.std(cropped[:, -1])) / 2

                    ## Find shape of pupil
                    # Threshold
                    thresholded = np.uint8(cv2.threshold(cropped, avg-(std*3), 255, cv2.THRESH_BINARY_INV)[1])
                    # Find contours
                    contours, heirarchy = cv2.findContours(thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    # if more than one contour
                    if len(contours) > 0:
                        # Get largest contour
                        largest_contour = max(contours, key=cv2.contourArea)
                        # sanity check size of largest contour
                        if(len(largest_contour) > 7):
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
                            print("Pupil size predicted by circles: {area1}".format(area1=darkest_circle_area))
                            print("Pupil Size predicted by ellipses: {area}".format(area=cv2.contourArea(largest_contour)))
                            pupil[f, 0] = darkest_circle[0]
                            pupil[f, 1] = darkest_circle[1]
                            pupil[f, 2] = cv2.contourArea(largest_contour)
                            # Fill debug displays and show
                            cv2.imshow("Eye", frame)
                            ret = cv2.waitKey(1)
                        else:
                            print("Pupil Size: n/a (too small)")
                            pupil[f,2] = -1
                    else:
                        print("Pupil Size: n/a (pupil off screen)")
                        pupil[f,2] = -2
                else:
                    print("Pupil Size: n/a (no contour)")
                    pupil[f,2] = -3
            else:
                print("Pupil Size: n/a (no circles)")
                pupil[f,2] = -4
            # Add current frame to average clip at correct slot
            day_avg_clip[:,:,f] = day_avg_clip[:,:,f] + gray
    # Save pupil size data
    print("Saving csv of positions and areas for {eye} eye...".format(eye=which_eye))
    padded_filename = which_eye + str(trial_number).zfill(4) + ".csv"
    csv_file = os.path.join(csv_path, padded_filename)
    np.savetxt(csv_file, pupil, fmt='%.2f', delimiter=',')
    # release video capture
    video.release()
    cv2.destroyAllWindows()

def save_average_clip_images(which_eye, no_of_frames, save_folder_path, images):
# Save images from trial clip to folder
    print("Saving averaged frames from {eye}...".format(eye=which_eye))
    for f in range(no_of_frames):

        # Create file name with padded zeros
        padded_filename = which_eye + str(f).zfill(4) + ".png"

        # Create image file path from save folder
        image_file_path = os.path.join(save_folder_path, padded_filename)

        # Extract gray frame from clip
        gray = np.uint8(images[:,:,f] * 255)

        # Write to image file
        ret = cv2.imwrite(image_file_path, gray)


### -------------------------------------------- ###
### LET THE ANALYSIS BEGIN!! ###

### log everything in a text file
# grab today's date
now = datetime.datetime.now()
# create log file named according to today's date
current_working_directory = os.getcwd()
log_filename = "log_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
log_file = os.path.join(current_working_directory, log_filename)
# set up log file to store all printed messages
sys.stdout = open(log_file, "w")

### ------------------------------------------- ###
# list all folders in Synology drive

# when working from Synology NAS drive
data_drive = r"\\Diskstation\SurprisingMinds"
# when working from local drive
#data_drive = r"C:\Users\KAMPFF-LAB-VIDEO\Documents\SurprisingMinds\fromSynology"

# get the subfolders, sort their names
data_folders = sorted(os.listdir(data_drive))
zipped_data = fnmatch.filter(data_folders, '*.zip')
zipped_names = [item[:-4] for item in zipped_data]

# figure out which days have already been analysed
analysed_drive = r"C:\Users\KAMPFF-LAB-VIDEO\Dropbox\SurprisingMinds\analysis\pythonWithAdam-csv"
analysed_folders = sorted(os.listdir(analysed_drive))
already_analysed = [item for item in zipped_names if item in analysed_folders]

# create dictionary of octopus clip start frames
octo_frames = {"stimuli024": 438, "stimuli025": 442, "stimuli026": 517, "stimuli027": 449, "stimuli028": 516, "stimuli029": 583}

# unzip each folder, do the analysis, skip #recycle aka data_folders[0]
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
    #analysis_folder = os.path.join(day_zipped[:-4], "Analysis")
    # when immediately placing analysis csv files in analysed drive
    analysis_folder = os.path.join(analysed_drive, item[:-4], "Analysis")

    # Analysis subfolders
    clip_folder = os.path.join(analysis_folder, "clip")
    csv_folder = os.path.join(analysis_folder, "csv")

    # Create analysis folder (and sub-folders) if it (they) does (do) not exist
    if not os.path.exists(analysis_folder):
        print("Creating analysis folder.")
        os.makedirs(analysis_folder)
    if not os.path.exists(clip_folder):
        print("Creating clip folder.")
        os.makedirs(clip_folder)
    if not os.path.exists(csv_folder):
        print("Creating csv folder.")
        os.makedirs(csv_folder)

    # create a temp folder in current working directory to store data (contents of unzipped folder)
    day_folder = os.path.join(current_working_directory, "temp")

    # unzip current zipped folder into temp folder, this function checks whether the folder is unzippable
    # if it unzips, the function returns True; if it doesn't unzip, the function returns False
    if unpack_to_temp(day_zipped, day_folder):

        # List all trial folders
        trial_folders = list_sub_folders(day_folder)
        num_trials = len(trial_folders)

        # Set temporal alignment parameters
        clip_length = 500

        # Allocate empty space for average frame and movie clip
        right_average_gray_clip = np.zeros((600,800,clip_length))
        left_average_gray_clip = np.zeros((600,800,clip_length))

        # Load all right eye movies and average
        current_trial = 0

        for trial_folder in trial_folders:
            # add exception handling so that a weird day doesn't totally break everything 
            try:
                # Load CSVs and create timestamps
                # ------------------------------
                print("Loading csv files...")
                # Get world movie timestamp csv path
                world_csv_path = glob.glob(trial_folder + '/*world.csv')[0]

                # Load world CSV
                world_timestamps = np.genfromtxt(world_csv_path, dtype=np.str, delimiter=' ')

                # Get eye timestamp csv paths
                right_eye_csv_path = glob.glob(trial_folder + '/*righteye.csv')[0]
                left_eye_csv_path = glob.glob(trial_folder + '/*lefteye.csv')[0]

                # Load eye CSVs
                right_eye_timestamps = np.genfromtxt(right_eye_csv_path, dtype=np.str, delimiter=' ')
                left_eye_timestamps = np.genfromtxt(left_eye_csv_path, dtype=np.str, delimiter=' ')

                # ------------------------------
                # Set temporary align frame to the frame counter closest to octopus_clip_start
                # octopus_clip_start here refers to frame when octopus appears, w.r.t. last frame of movie
                stimuli_number = world_csv_path.split("_")[-2]
                octopus_start_frame = octo_frames[stimuli_number]
                print("Finding octopus in video {name}...".format(name=stimuli_number))
                # ------------------------------
                print("Checking octopus...")
                # Get world video filepath
                world_video_path = glob.glob(trial_folder + '/*world.avi')[0]

                # Open world video
                world_video = cv2.VideoCapture(world_video_path)

                # Jump to start of when octopus video clip starts (position)
                ret = world_video.set(cv2.CAP_PROP_POS_FRAMES, octopus_start_frame)
                
                # Show the frame to check for start of octopus clip (ground truth)
                ret, frame = world_video.read()
                plt.imshow(frame)
                plt.show(block=False)
                plt.pause(1)
                plt.close()
                
                # Set frame in world video where octopus appears
                world_octopus_frame = octopus_start_frame
                print("Octopus clip begins at frame {number} in world video".format(number=world_octopus_frame))

                # ------------------------------
                # Find align frame for analyzing eye videos, aka clip_offset + start of octopus clip
                # this means our analysis starts a little bit (# of frames = clip_offset) before the start of the octopus clip
                right_eye_octopus = find_target_frame(world_timestamps, right_eye_timestamps, world_octopus_frame)
                left_eye_octopus = find_target_frame(world_timestamps, left_eye_timestamps, world_octopus_frame)
                # ------------------------------
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
                find_pupil("right", current_trial, right_video_path, right_eye_octopus, clip_length, right_average_gray_clip, csv_folder)
                # Find left eye pupils and save pupil data
                print("Finding left eye pupils...")
                find_pupil("left", current_trial, left_video_path, left_eye_octopus, clip_length, left_average_gray_clip, csv_folder)
                
                # Report progress
                world_video.release()
                cv2.destroyAllWindows()
                print("Finished Trial: {trial}".format(trial=current_trial))
                current_trial = current_trial + 1
            except Exception: 
                cv2.destroyAllWindows()
                print("Trial {trial} failed!".format(trial=current_trial))
                current_trial = current_trial + 1

        # Compute average clips
        right_average_gray_clip = (right_average_gray_clip/num_trials)/255.0
        left_average_gray_clip = (left_average_gray_clip/num_trials)/255.0

        # Save averaged images from day to clip folder
        save_average_clip_images("right", clip_length, clip_folder, right_average_gray_clip)
        save_average_clip_images("left", clip_length, clip_folder, left_average_gray_clip)

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