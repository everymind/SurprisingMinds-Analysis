import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

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
#daily_folders = daily_folders[0:15]
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
        target_pts = np.vstack((target_x, target_y)).T
        calib_pts = np.vstack((calib_x, calib_y)).T
        h, status = cv2.findHomography(target_pts, calib_pts)

        # Transform gazes
        gaze_x = np.zeros(5)
        gaze_y = np.zeros(5)
        for i in range(5):
            pupil_pt = np.array([target_x[i],target_y[i],1])
            result_pt_homg = np.dot(h, pupil_pt)
            gaze_pt = result_pt_homg / result_pt_homg[2]
            gaze_x[i] = gaze_pt[0]
            gaze_y[i] = gaze_pt[1] 

        # Save gaze per trial...
        gaze = np.zeros((num_samples, 2))
        for i in range(num_samples):
            pupil_pt = np.array([x[i], y[i], 1])
            result_pt_homg = np.dot(h, pupil_pt)
            gaze[i,0] = result_pt_homg[0] / result_pt_homg[2]
            gaze[i,1] = result_pt_homg[1] / result_pt_homg[2]
        
        # Store Gaze
        output_path = data_folder + '/gazes/%d_%s_gaze_%d.data' % (stimulus, eye, trial_count)
        gaze = np.float32(gaze)
        gaze.tofile(output_path)
        trial_count = trial_count + 1

        # Plot
        plot = False
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
if plot:
    plt.show()

# Load Gaze files
gaze_folder = data_folder + '/gazes'
gaze_files = glob.glob(gaze_folder + '/*.data')

# Make movie container
movie = np.zeros((40, 40, 1000), dtype=np.float32)
for gaze_file in gaze_files:

    # Get stimulus number
    trial_name = os.path.basename(gaze_file)
    fields = trial_name.split(sep='_')
    eye = fields[1]
    stimulus = int(fields[0])
    if(stimulus < 5):
        gaze_flat = np.fromfile(gaze_file, dtype=np.float32)

        # Filter for valid screen positions
        gaze_filtered = np.copy(gaze_flat)
        gaze_filtered[(gaze_flat < -0.999) + (gaze_flat > +0.999)] = np.nan
        
        # Reshpae to 2D array
        num_samples = np.int(len(gaze_filtered) / 2)
        gaze = np.reshape(gaze_filtered, (num_samples, -1))

        # Bin to 40 ms (reshape and average across columns)
        gaze_x_binned = gaze[:10000, 0] 
        gaze_x_binned = np.nanmean(np.reshape(gaze_x_binned, (1000,10)),1)
        gaze_y_binned = gaze[:10000, 1] 
        gaze_y_binned = np.nanmean(np.reshape(gaze_y_binned, (1000,10)),1)

        # Fill in movie frames
        for f in range(1000):
            if (np.isnan(gaze_x_binned[f]) or np.isnan(gaze_y_binned[f])):
                continue
            x_pixel = np.int(np.floor(20 * (gaze_x_binned[f] + 1.0)))
            y_pixel = np.int(np.floor(20 * (gaze_y_binned[f] + 1.0)))
            movie[y_pixel, x_pixel, f] = movie[y_pixel, x_pixel, f] + 1

# Save movie frames
folder = '/home/kampff/Desktop/frames'
for f in range(1000):
    filename = folder + '/' + str(10000+f) + '.png'
    cv2.imwrite(filename, movie[:,:, f])

#FIN