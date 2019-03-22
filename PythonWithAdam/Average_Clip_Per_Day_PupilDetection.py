import os
import glob
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import shutil
import fnmatch

### FUNCTIONS ###

def unpack_to_temp(path_to_zipped, path_to_temp):
    try:
        # unzip the folder
        print("Unzipping files in {folder}...".format(folder=path_to_zipped))
        day_unzipped = zipfile.ZipFile(path_to_zipped, mode="r")
        # extract files into temp folder
        day_unzipped.extractall(path_to_temp)
        # close the unzipped file
        day_unzipped.close()
        print("Finished unzipping {folder}!".format(folder=path_to_zipped))
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

def find_target_frame(ref_timestamps_csv, target_timestamps_csv, start_frame, target_time):
    # Find the frame that best matches the target time (seconds) from the end of a movie file
    # Get last frame time
    start_timestamp = ref_timestamps_csv[start_frame]
    start_timestamp = start_timestamp.split('+')[0][:-1]
    start_time = datetime.datetime.strptime(start_timestamp, "%Y-%m-%dT%H:%M:%S.%f")

    # Generate delta times (w.r.t. last frame) for every frame timestamp
    frame_counter = 0
    for timestamp in target_timestamps_csv:
        timestamp = timestamp.split('+')[0][:-1]
        time = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f")
        timedelta = start_time - time
        seconds_until_alignment = timedelta.total_seconds()
        if((seconds_until_alignment - target_time) < 0):
            break
        frame_counter = frame_counter + 1
    return frame_counter

def look_for_octopus(video):
    # Loop through frames of world video and check for octopus (ground truth)
    for f in range(200):
        # Read frame at current position
        ret, frame = video.read()
        # Convert to grayscale
        gray = np.mean(frame,2)
        # Measure ROI intensity
        roi = gray[51:58, 63:70]
        intensity = np.mean(np.mean(roi))
        # Is there an octopus?
        if(intensity > 100):
            break
    return f

def find_pupil(which_eye, trial_number, video_path, align_frame, no_of_frames, day_avg_clip):
    # Loop through frames of eye video to find and save pupil xy positon and area
    # Loop through frames, greyscale them, and add each frame to day_avg_clip: this generates
    #   a frame that is the average of every trial in the same day

    # Open right eye video
    video = cv2.VideoCapture(video_path)
    # Jump to specific frame (position) for alignment purposes 
    ret = video.set(cv2.CAP_PROP_POS_FRAMES, align_frame)

    # Open display window for debugging
    cv2.namedWindow("Eye")
    # Create empty data array
    pupil = np.zeros((clip_length, 3))

    # Loop through frames of eye video to find and save pupil xy positon and area
    for f in range(0, no_of_frames, 1):
        
        # Read frame at current position
        ret, frame = video.read()

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
                                    param1=75, param2=30,
                                    minRadius=16, maxRadius=200)

            # If there are no circles, then what??
            if circles is not None:

                # If there are more than one, take the darkest

                # Using the best circle...crop around center
                # Threshold
                # Fit an ellipse

                # Crop
                eye_circle = np.uint16(np.around(circles[0][0]))
                left = eye_circle[0] - 64
                top = eye_circle[1] - 64
                crop_size = 128

                # Check boundarys of image
                if( (left >= 0) and (top >= 0) and ((left + crop_size) < 800) and ((top + crop_size) < 600) ):

                    cropped = gray[top:(top + crop_size), left:(left+crop_size)]
                    
                    # Compute average and stdev of all pixel luminances along border
                    avg = (np.mean(cropped[:, 0]) + np.mean(cropped[:, -1])) / 2
                    std = (np.std(cropped[:, 0]) + np.std(cropped[:, -1])) / 2

                    # Threshold
                    thresholded = np.uint8(cv2.threshold(cropped, avg-(std*3), 255, cv2.THRESH_BINARY_INV)[1])

                    # Find contours
                    contours, heirarchy = cv2.findContours(thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                    if len(contours) > 0:
                        # Get largest contour
                        largest_contour = max(contours, key=cv2.contourArea)

                        if(len(largest_contour) > 7):

                            # Fit ellipse to largest contour
                            ellipse = cv2.fitEllipse(largest_contour)
                        
                            # Shift ellipse back to full frame coordinates
                            shifted_center = (np.int(ellipse[0][0]) + left, np.int(ellipse[0][1]) + top)

                            # Draw circles
                            if circles is not None:
                                circles = np.uint16(np.around(circles))
                                for i in circles[0, :]:
                                    center = (i[0], i[1])
                                    # circle center
                                    cv2.circle(frame, center, 5, (0, 100, 100), 1)
                                    # circle outline
                                    radius = i[2]
                                    cv2.circle(frame, center, radius, (255, 0, 255), 1)

                            # Draw ellipse
                            axes = (np.int(ellipse[1][0]/2),np.int(ellipse[1][1]/2)) 
                            angle = np.int(ellipse[2])
                            frame = cv2.ellipse(frame, shifted_center, axes, angle, 0, 360, (0, 255, 0), 5, cv2.LINE_AA, 0)

                            # Save Data
                            print("Pupil Size: {area}".format(area=cv2.contourArea(largest_contour)))
                            pupil[f, 0] = shifted_center[0]
                            pupil[f, 1] = shifted_center[1]
                            pupil[f, 2] = cv2.contourArea(largest_contour)

                            # Fill debug display and show
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
    padded_filename = str(trial_number).zfill(4)
    csv_file = os.path.join(csv_folder, which_eye, padded_filename, ".csv")
    np.savetxt(csv_file, pupil, fmt='%.2f', delimiter=',')
    # release video capture
    video.release()
    cv2.destroyAllWindows()

def save_average_clip_images(which_eye, no_of_frames, save_folder_path, images):
# Save images from trial clip to folder
    print("Saving averaged frames...")
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

# list all folders in Synology drive
data_drive = r"\\Diskstation\SurprisingMinds"
data_folders = sorted(os.listdir(data_drive))
zipped_data = fnmatch.filter(data_folders, '*.zip')
already_analysed = [item for item in data_folders if item not in zipped_data]

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
    analysis_folder = os.path.join(day_zipped[:-4], "Analysis")
    clip_folder = os.path.join(analysis_folder, "clip")
    csv_folder = os.path.join(analysis_folder, "csv")

    # Create analysis folder (and sub-folders) if it (they) does (do) not exist
    if not os.path.exists(analysis_folder):
        print("Creating analysis folder.")
        os.makedirs(analysis_folder)
    if not os.path.exists(clip_folder):
        os.makedirs(clip_folder)
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)

    # create a temp folder to store data (contents of unzipped folder)
    day_folder = os.path.join(data_drive, "temp")
    # unzip current zipped folder into temp folder, this function checks whether the folder is unzippable
    # if it unzips, the function returns True; if it doesn't unzip, the function returns False
    if unpack_to_temp(day_zipped, day_folder):

        # List all trial folders
        trial_folders = list_sub_folders(day_folder)
        num_trials = len(trial_folders)

        # Set temporal alignment parameters
        octopus_clip_start = 18.3 # seconds w.r.t. end of world movie
        octopus_appear_time = 12 # seconds w.r.t. end of world movie
        clip_length = 500
        clip_offset = -120

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
                octopus_appears_frame = find_target_frame(world_timestamps, world_timestamps, -1, octopus_clip_start) 
                # ------------------------------
                # Get world video filepath
                world_video_path = glob.glob(trial_folder + '/*world.avi')[0]
                # Open world video
                world_video = cv2.VideoCapture(world_video_path)
                # Jump to start of when octopus appears (position)
                ret = world_video.set(cv2.CAP_PROP_POS_FRAMES, octopus_appears_frame)
                # Loop through frames of world video and check for octopus (ground truth)
                frame_adjust = look_for_octopus(world_video)
                # Set frame in world video where octopus appears
                world_octopus_frame = octopus_appears_frame + frame_adjust
                # ------------------------------
                # Find align frame for analyzing eye videos, aka clip_offset + start of octopus clip
                # this means our analysis starts a little bit (# of frames = clip_offset) before the start of the octopus clip
                right_eye_octopus = find_target_frame(world_timestamps, right_eye_timestamps, world_octopus_frame, 0)
                right_align_frame = right_eye_octopus + clip_offset
                left_eye_octopus = find_target_frame(world_timestamps, left_eye_timestamps, world_octopus_frame, 0)
                left_align_frame = left_eye_octopus + clip_offset
                # ------------------------------
                # ------------------------------
                # ------------------------------
                # Now start pupil detection
                # Create "eye kernel"
                ### DO WE STILL NEED THIS?? ###
                eye_size = 64
                surround_size = 128
                eye_kernel = np.zeros((surround_size, surround_size), dtype=np.float32)
                eye_kernel = cv2.circle(eye_kernel, (eye_size, eye_size), np.int(eye_size / 2), (1,), -1)
                eye_kernel = -1 * (eye_kernel - np.mean(np.mean(eye_kernel)))
                
                # ------------------------------
                # Get right eye video filepath
                right_video_path = glob.glob(trial_folder + '/*righteye.avi')[0]
                # Get left eye video filepath
                left_video_path = glob.glob(trial_folder + '/*lefteye.avi')[0]
                # Find right eye pupils and save pupil data
                print("Finding right eye pupils...")
                find_pupil("right", current_trial, right_video_path, right_align_frame, clip_length, right_average_gray_clip)
                # Find left eye pupils and save pupil data
                print("Finding left eye pupils...")
                find_pupil("left", current_trial, left_video_path, left_align_frame, clip_length, left_average_gray_clip)
                # Report progress
                print("Finished Trial: {trial}".format(trial=current_trial))
                current_trial = current_trial + 1
            except Exception: 
                print("Trial {trial} failed!".format(trial=current_trial))
                current_trial = current_trial + 1

        # Compute average clips
        right_average_gray_clip = (right_average_gray_clip/num_trials)/255.0
        left_average_gray_clip = (left_average_gray_clip/num_trials)/255.0

        # Save averaged images from day to clip folder
        save_average_clip_images("right", clip_length, clip_folder, right_average_gray_clip)
        save_average_clip_images("left", clip_length, clip_folder, left_average_gray_clip)

        # report progress
        video.release()
        cv2.destroyAllWindows()
        print("Finished {day}".format(day=day_zipped[:-4]))

        # delete temporary file with unzipped data contents
        shutil.rmtree(day_folder)

#FIN
