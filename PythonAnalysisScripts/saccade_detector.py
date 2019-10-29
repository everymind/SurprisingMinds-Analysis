import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# TO DO:
# Save each trials "peaks" as a spereate file (like is done with speeds)
# funcitons?

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

        # Measure "speed" (change in x and y)
        dx = np.diff(x, prepend=[0])
        dy = np.diff(y, prepend=[0])
        speed = np.sqrt(dx*dx + dy*dy)
        speed = np.float32(speed)

        # Store
        output_path = data_folder + '/speeds/%d_%s_speed_%d.data' % (stimulus, eye, trial_count)
        speed.tofile(output_path)
        trial_count = trial_count + 1

        # Plot
        plot = False
        if plot:
            plt.figure()
            plt.subplot(2,2,1)
            plt.plot(raw_x)
            plt.plot(x)
            plt.subplot(2,2,2)
            plt.plot(raw_y)
            plt.plot(y)
            plt.subplot(2,2,3)
            plt.plot(raw_area)
            plt.plot(area)
            plt.subplot(2,2,4)
            plt.plot(speed)
            plt.show()

        # Report progress
        print(trial_count)


# Load speed files
speed_folder = data_folder + '/speeds'
speed_files = glob.glob(speed_folder + '/*.data')
num_files = len(speed_files)
peak_raster = np.zeros((num_files, 14000))
speed_raster = np.zeros((num_files, 14000))

window_size = 50
all_peak_windows = np.zeros((0,window_size))
all_peak_speeds = np.zeros(0)
all_peak_durations = np.zeros(0)
all_peak_intervals = np.zeros(0)
count = 0

plt.figure()
for s in range(6):
    for i, speed_file in enumerate(speed_files):

        # Get stimulus number
        trial_name = os.path.basename(speed_file)
        fields = trial_name.split(sep='_')
        eye = fields[1]
        stimulus = int(fields[0])

        # Check if current stimulus number
        if(stimulus == s):

            # Load speed_file
            speed = np.fromfile(speed_file, dtype=np.float32)
            speed_raster[count, :len(speed)] = speed > 1.25

            # Find "peaks" greater than some threshold?
            low_threshold = 0.5
            high_threshold = 1.5
            peak_start_times = []
            peak_stop_times = []
            peaking = False
            for i, sp in enumerate(speed):
                # Look for a new peak
                if(not peaking):
                    if(sp > high_threshold):
                        peaking = True
                        peak_start_times.append(i)
                # Track ongoing peak    
                else:
                    if(sp < low_threshold):
                        peaking = False       
                        peak_stop_times.append(i)

            # Convert to arrays
            peak_start_times = np.array(peak_start_times)
            peak_stop_times = np.array(peak_stop_times)

            # Throw out the first peak
            peak_start_times = peak_start_times[1:]
            peak_stop_times = peak_stop_times[1:]

            # Throw out last peak if incomplete
            if len(peak_start_times) > len(peak_stop_times):
                peak_start_times = peak_start_times[:-1]

            # Find peak durations
            peak_durations = peak_stop_times - peak_start_times

            # Find peak speed and indices
            peak_speeds = []
            peak_indices = []
            for start, stop in zip(peak_start_times,peak_stop_times):
                peak_speed = np.max(speed[start:stop])
                peak_index = np.argmax(speed[start:stop])
                peak_speeds.append(peak_speed)
                peak_indices.append(start + peak_index)

            # Convert to arrays
            peak_speeds = np.array(peak_speeds)
            peak_indices = np.array(peak_indices)

            # Measure inter-peak_interval
            peak_intervals = np.diff(peak_indices, prepend=[0])

            # Filter for good saccades
            good_peaks = (peak_intervals > 25) * (peak_durations < 30) * (peak_durations > 4) * (peak_speeds < 100)
            peak_speeds = peak_speeds[good_peaks]
            peak_indices = peak_indices[good_peaks]
            peak_durations = peak_durations[good_peaks]
            peak_intervals = peak_intervals[good_peaks]

            # Extract windows around peak maxima
            for peak_index in peak_indices:
                left_border = np.int(peak_index - np.round(window_size/2))
                right_border = np.int(left_border + window_size)

                # Check that window fits
                if left_border < 0:
                    continue
                if right_border > len(speed):
                    continue

                peak_window = speed[left_border:right_border]
                all_peak_windows = np.vstack((all_peak_windows, peak_window))

            # Make some peak categories
            big_speeds = (peak_speeds < 75) * (peak_speeds > 45)
            med_speeds = (peak_speeds < 25) * (peak_speeds > 15)
            lil_speeds = (peak_speeds < 15) * (peak_speeds > 1)

            # Plot a saccade raster
            num_peaks = np.sum(big_speeds)
            row_value = count*np.ones(num_peaks)
            plt.subplot(3,1,1)
            plt.plot(peak_indices[big_speeds], row_value, 'r.', alpha=0.01)

            num_peaks = np.sum(med_speeds)
            row_value = count*np.ones(num_peaks)
            plt.subplot(3,1,2)
            plt.plot(peak_indices[med_speeds], row_value, 'b.', alpha=0.01)

            num_peaks = np.sum(lil_speeds)
            row_value = count*np.ones(num_peaks)
            plt.subplot(3,1,3)
            plt.plot(peak_indices[lil_speeds], row_value, 'k.', alpha=0.1)

            # Add to all "peak" arrays
            all_peak_speeds = np.hstack((all_peak_speeds, peak_speeds))
            all_peak_durations = np.hstack((all_peak_durations, peak_durations))
            all_peak_intervals = np.hstack((all_peak_intervals, peak_intervals))

            # Store peaks in peak raster
            peak_raster[count, peak_intervals] = 1

            # Report
            print(count)
            count = count + 1
plt.show()

# PCA (or rather SVD)
u, s, vh = np.linalg.svd(all_peak_windows, full_matrices=False)
plt.plot(vh[0,:], 'r')
plt.plot(vh[1,:], 'b')
plt.plot(vh[2,:], 'g')
plt.show()

# Project all my peak windows into PC1 and PC2
PC1 = -vh[0,:]
PC2 = -vh[2,:]
all_prj_1 = np.dot(all_peak_windows, PC1)
all_prj_2 = np.dot(all_peak_windows, PC2)
plt.plot(all_prj_1, all_prj_2, 'k.', alpha=0.1)
plt.show()

plt.plot(s, '.')
plt.show()

# Meak average peak
mean_peak = np.mean(all_peak_windows, 0)
plt.plot(all_peak_windows[:1000, :].T, 'r', alpha=0.01)
plt.plot(mean_peak)
plt.show()

# Average
mean_raster = np.mean(speed_raster, 0)
plt.plot(mean_raster)
plt.show()

# Bin
binned_raster = np.mean(np.reshape(speed_raster, (count, 2800,-1)),2)
plt.imshow(binned_raster, cmap='gray')
plt.show()

# Plot all the peak params
plt.figure()
plt.plot(all_peak_speeds, all_peak_durations, 'k.', alpha=0.01)
plt.show()

plt.figure()
plt.plot(all_peak_speeds, all_peak_intervals, 'k.', alpha=0.01)
plt.show()

#FIN