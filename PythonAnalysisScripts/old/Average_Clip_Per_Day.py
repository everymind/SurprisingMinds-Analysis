import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# List relevant data locations
root_folder = "/home/kampff/DK"
day_folder = root_folder + "/SurpriseIntelligence_2017-07-25"
save_folder = root_folder + "/Clip"

# List all trial folders
trial_folders = []
for folder in os.listdir(day_folder):
    if(os.path.isdir(day_folder + '/' + folder)):
        trial_folders.append(day_folder + '/' + folder)
num_trials = len(trial_folders)

# Set averaging parameters
align_frame = 900
clip_length = 10

# Allocate empty space for average frame and movie clip
average_grayscale_clip = np.zeros((600,800,clip_length))

# Load all left eye movies and average
current_trial = 0
for trial_folder in trial_folders:

    # Get video filepath
    left_video_path = glob.glob(trial_folder + '/*lefteye.avi')[0]

    # Open video
    video = cv2.VideoCapture(left_video_path)

    # Jump to specific frame (position) for alignment purposes (currently arbitrary)
    ret = video.set(cv2.CAP_PROP_POS_FRAMES, align_frame)

    # Loop through frames for clip extraction
    for f in range(clip_length):
        
        # Read frame at current position
        ret, frame = video.read()

        # Convert to grayscale
        gray = np.mean(frame,2)

        # Add current frame to average clip at correct slot
        average_grayscale_clip[:,:,f] = average_grayscale_clip[:,:,f] + gray

    # Report progress
    print(current_trial)
    current_trial = current_trial + 1

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
