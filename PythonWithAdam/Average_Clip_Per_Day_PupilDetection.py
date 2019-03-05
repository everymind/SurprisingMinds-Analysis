import os
import glob
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt

# List relevant data locations
root_folder = "/home/kampff/DK"
day_folder = root_folder + "/SurpriseIntelligence_2017-07-25"
save_folder = root_folder + "/Clip"
csv_folder = root_folder + "/csv"

# List all trial folders
trial_folders = []
for folder in os.listdir(day_folder):
    if(os.path.isdir(day_folder + '/' + folder)):
        trial_folders.append(day_folder + '/' + folder)
num_trials = len(trial_folders)

# Set averaging parameters
align_time = 18.3 # w.r.t. end of world movie
clip_length = 360
clip_offset = -120

# Allocate empty space for average frame and movie clip
average_grayscale_clip = np.zeros((600,800,clip_length))

# Load all right eye movies and average
current_trial = 0
for trial_folder in trial_folders:




    # Load CSV and create timestamps
    # ------------------------------

    # Get world movie timestamp csv path
    world_csv_path = glob.glob(trial_folder + '/*world.csv')[0]

    # Load world CSV
    world_timestamps = np.genfromtxt(world_csv_path, dtype=np.str, delimiter=' ')

    # Get eye (right) timestamp csv path
    right_eye_csv_path = glob.glob(trial_folder + '/*righteye.csv')[0]

    # Load eye (right) CSV
    right_eye_timestamps = np.genfromtxt(right_eye_csv_path, dtype=np.str, delimiter=' ')

    # ------------------------------

    # Get last frame time
    last_timestamp = world_timestamps[-1]
    last_timestamp = last_timestamp.split('+')[0][:-1]
    last_time = datetime.datetime.strptime(last_timestamp, "%Y-%m-%dT%H:%M:%S.%f")

    # Generate delta times (w.r.t. last frame) for every frame timestamp
    frame_counter = 0
    for timestamp in world_timestamps:
        timestamp = timestamp.split('+')[0][:-1]
        time = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f")
        timedelta = last_time - time
        seconds_until_end = timedelta.total_seconds()
        if((seconds_until_end - align_time) < 0):
            break
        frame_counter = frame_counter + 1

    # Set temporary align frame to the frame counter closest to align_time
    temp_align_frame = frame_counter

    # Get world video filepath
    world_video_path = glob.glob(trial_folder + '/*world.avi')[0]
    
    # Open world video
    video = cv2.VideoCapture(world_video_path)

    # Jump to temprary alignment frame (position)
    ret = video.set(cv2.CAP_PROP_POS_FRAMES, temp_align_frame)

    # Loop through frames and look for octopus
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
    
    # Set world align frame
    world_align_frame = temp_align_frame + f
    world_align_frame = world_align_frame

    # Set world align time
    world_align_timestamp = world_timestamps[world_align_frame]
    world_align_timestamp = world_align_timestamp.split('+')[0][:-1]
    world_align_time = datetime.datetime.strptime(world_align_timestamp, "%Y-%m-%dT%H:%M:%S.%f")

    # Find (right) eye align frame

    # Generate delta times (w.r.t. world_align_time) for every (right) eye frame timestamp
    frame_counter = 0
    for timestamp in right_eye_timestamps:
        timestamp = timestamp.split('+')[0][:-1]
        time = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f")
        timedelta = world_align_time - time
        seconds_until_alignment = timedelta.total_seconds()
        if(seconds_until_alignment < 0):
            break
        frame_counter = frame_counter + 1

    # Find (right) eye align frame
    align_frame = frame_counter + clip_offset





    # Now start "spatial" alignment (i.e pupil detection)



    # Get (right) eye video filepath
    right_video_path = glob.glob(trial_folder + '/*righteye.avi')[0]
    
    # Open video
    video = cv2.VideoCapture(right_video_path)

    # Jump to specific frame (position) for alignment purposes (currently arbitrary)
    ret = video.set(cv2.CAP_PROP_POS_FRAMES, align_frame)

    # Create "eye kernel"
    eye_size = 64
    surround_size = 128
    eye_kernel = np.zeros((surround_size, surround_size), dtype=np.float32)
    eye_kernel = cv2.circle(eye_kernel, (eye_size, eye_size), np.int(eye_size / 2), (1,), -1)
    eye_kernel = -1 * (eye_kernel - np.mean(np.mean(eye_kernel)))

    # Open display window for debugging
    cv2.namedWindow("Eye")

    # Create empty data array
    pupil_areas = np.zeros(clip_length)

    # Loop through frames for clip extraction
    for f in range(0, clip_length, 1):
        
        # Read frame at current position
        ret, frame = video.read()


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
                image_out, contours, heirarchy = cv2.findContours(thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                if len(contours) > 0:
                    # Get largest contour
                    largest_contour = max(contours, key=cv2.contourArea)

                    if(len(largest_contour) > 7):

                        # Save Data
                        print("Pupil Size: {area}".format(area=cv2.contourArea(largest_contour)))
                        pupil_areas[f] = cv2.contourArea(largest_contour)

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

                        # Fill debug display and show
                        cv2.imshow("Eye", frame)
                        ret = cv2.waitKey(1)
                    else:
                        print("Pupil Size: n/a (too small)")
                        pupil_areas[f] = -1
                else:
                    print("Pupil Size: n/a (pupil off screen)")
                    pupil_areas[f] = -2
            else:
                print("Pupil Size: n/a (no contour)")
                pupil_areas[f] = -3
        else:
            print("Pupil Size: n/a (no circles)")
            pupil_areas[f] = -4

        # Add current frame to average clip at correct slot
        average_grayscale_clip[:,:,f] = average_grayscale_clip[:,:,f] + gray

    # Save pupil size data
    padded_filename = str(current_trial).zfill(4)
    csv_file = csv_folder + "/" + padded_filename + ".csv"
    np.savetxt(csv_file, pupil_areas, fmt='%.2f', delimiter=',')
    
    # Report progress
    print("Finished Trial: {trial}".format(trial=current_trial))
    current_trial = current_trial + 1
    cv2.destroyAllWindows()

# Compute average clip
average_grayscale_clip = (average_grayscale_clip/num_trials)/255.0

# Save images from clip to folder
for f in range(clip_length):

    # Create file name with padded zeros
    padded_filename = str(f).zfill(4)

    # Create image file path from save folder
    image_file_path = save_folder + "/" + padded_filename + ".png" 

    # Extract gray frame from clip
    gray = np.uint8(average_grayscale_clip[:,:,f] * 255)

    # Write to image file
    ret = cv2.imwrite(image_file_path, gray)

#FIN
