import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Specify data folder
data_folder = '/home/kampff/Data/Surprising'

# Find daily folders
daily_folders = glob.glob(data_folder + '/*')
num_days = len(daily_folders)

# Count number of files
num_files = 0
for df, daily_folder in enumerate(daily_folders):

    # Find csv paths (platform independent)
    csv_paths = glob.glob(daily_folder + '/analysis/csv/*.csv')
    if(len(csv_paths) == 0):
        csv_paths = glob.glob(daily_folder + '/Analysis/csv/*.csv')
    num_files = len(csv_paths) + num_files
print(num_files)

# Process all daily folders
trial_count = 0
daily_folders = daily_folders[0:15]
plt.figure()
for df, daily_folder in enumerate(daily_folders):

    # Find csv paths
    csv_paths = glob.glob(daily_folder + '/analysis/csv/*.csv')
    if(len(csv_paths) == 0):
        csv_paths = glob.glob(daily_folder + '/Analysis/csv/*.csv')
    num_files = len(csv_paths)

    # Process all csv files in a folder
    for cp, csv_path in enumerate(csv_paths):

        # Extract eye name and stimulus number
        trial_name = os.path.basename(csv_path)
        fields = trial_name.split(sep='_')
        eye = fields[0]
        stimulus = int(fields[1][-1:])-4

        # Load data
        data = np.genfromtxt(csv_path, delimiter=',')
        raw_x = data[:,0]
        raw_y = data[:,1]
        raw_area = data[:,2]
        x = np.copy(raw_x)
        y = np.copy(raw_y)
        area = np.copy(raw_area)
        num_samples = len(x)

        # Extract valid X and Y values
        good_indices = np.where(area > 0)[0]

        # Exclude crappy trials
        if(len(good_indices) < 200):
            break
        good_x = x[good_indices]
        good_y = y[good_indices]
        good_area = area[good_indices]
        num_valid = len(good_indices)

        # Start with first valid values
        if x[0] < 0:
            x[0] = good_x[0]
            y[0] = good_y[0]
            area[0] = good_area[0]

        # Interpolate X and Y values across tracking errors/empty frames
        count = 1
        for i in range(1, num_valid):
            next_valid_index = good_indices[i]
            next_valid_x = good_x[i]
            next_valid_y = good_y[i]
            next_valid_area = good_area[i]
            step_count = (next_valid_index - count + 1)
            step_x = (next_valid_x - x[count - 1]) / step_count
            step_y = (next_valid_y - y[count - 1]) / step_count
            step_area = (next_valid_area - area[count - 1]) / step_count
            for j in range(step_count):
                x[count] = x[count - 1] + step_x
                y[count] = y[count - 1] + step_y
                area[count] = area[count - 1] + step_area
                count += 1

        # By now, we have X, Y, and Area for every time bucket (linearly interpolated)

        # Smooth (8 time-buckets: ~ 32 ms, 30 Hz)
        smooth_kernel = np.ones(8) / 8
        x = np.convolve(x, smooth_kernel, mode='same')
        y = np.convolve(y, smooth_kernel, mode='same')
        area = np.convolve(area, smooth_kernel, mode='same')

        # Get pupil position at known "calibration" coordinates

        # Set calibration positions
        calib_x = np.zeros(5)
        calib_y = np.zeros(5)
        calib_x[0] = -1
        calib_y[0] = +1

        calib_x[1] = +1
        calib_y[1] = -1

        calib_x[2] = -1
        calib_y[2] = -1

        calib_x[3] = +1
        calib_y[3] = +1

        calib_x[4] = 0
        calib_y[4] = 0

        # Calibration time buckets
        calib_shift = 50
        calib_window = 50
        calib_times = np.zeros(5, dtype=np.int)
        calib_times[0] = 1126 + calib_shift
        calib_times[1] = 1798 + calib_shift
        calib_times[2] = 2469 + calib_shift
        calib_times[3] = 3102 + calib_shift
        calib_times[4] = 3784 + calib_shift
        
        # Measure calibrated Gaze
        target_x = np.zeros(5)
        target_y = np.zeros(5)
        for i in range(5):
            target_x[i] = np.nanmean(x[calib_times[i]:(calib_times[i] + calib_window)])
            target_y[i] = np.nanmean(y[calib_times[i]:(calib_times[i] + calib_window)])        

        # Compute homography (map between pupil coordinates and screen position)
        left = (target_x[0] + target_x[2]) / 2
        right = (target_x[1] + target_x[3]) / 2
        top = (target_y[0] + target_y[3]) / 2
        bottom = (target_y[1] + target_y[2]) / 2
        width = right-left
        height = top-bottom

        # Transform gazes
        gaze_x = np.zeros(5)
        gaze_y = np.zeros(5)
        for i in range(5):
            gaze_x[i] = (2 * ((target_x[i] - left) / width)) - 1 
            gaze_y[i] = (2 * ((target_y[i] - bottom) / height)) - 1 

        # Better (parrelogram like model interpolation)

        # Save gaze per trial...
        


        # Store Gaze
        output_path = data_folder + '/gazes/%d_%s_gaze_%d.data' % (stimulus, eye, trial_count)
        gaze.tofile(output_path)
        trial_count = trial_count + 1

        # Plot
        plot = True
        if plot:
            if(eye == 'left'):
                plt.subplot(2,2,1)
            else:
                plt.subplot(2,2,2)
            plt.plot(target_x[0], target_y[0], 'r.', alpha=0.1)
            plt.plot(target_x[1], target_y[1], 'b.', alpha=0.1)
            plt.plot(target_x[2], target_y[2], 'm.', alpha=0.1)
            plt.plot(target_x[3], target_y[3], 'c.', alpha=0.1)
            plt.plot(target_x[4], target_y[4], 'g.', alpha=0.1)
            if(eye == 'left'):
                plt.subplot(2,2,3)
            else:
                plt.subplot(2,2,4)
            plt.plot(gaze_x[0], gaze_y[0], 'r.', alpha=0.1)
            plt.plot(gaze_x[1], gaze_y[1], 'b.', alpha=0.1)
            plt.plot(gaze_x[2], gaze_y[2], 'm.', alpha=0.1)
            plt.plot(gaze_x[3], gaze_y[3], 'c.', alpha=0.1)
            plt.plot(gaze_x[4], gaze_y[4], 'g.', alpha=0.1)

        # Report progress
        print(trial_count)

# Show figure
plt.show()


# Load Gaze files

#FIN