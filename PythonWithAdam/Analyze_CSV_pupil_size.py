import os
import glob
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

### FUNCTIONS ###
def load_daily_pupil_areas(which_eye, day_folder_path, max_trial_length, bucket_size, sample_rate_ms): 
    if (sample_rate_ms % bucket_size == 0):
        bucket_window = int(sample_rate_ms/bucket_size)
        trial_length = math.ceil(max_trial_length/bucket_window)
        # List all csv trial files
        trial_files = glob.glob(day_folder_path + os.sep + which_eye + "*.csv")
        num_trials = len(trial_files)

        data_contours = np.empty((num_trials, trial_length))
        data_contours[:] = -6

        data_circles = np.empty((num_trials, trial_length))
        data_circles[:] = -6

        index = 0
        for trial_file in trial_files:
            trial_name = trial_file.split(os.sep)[-1]
            trial = np.genfromtxt(trial_file, dtype=np.float, delimiter=",")
            no_of_samples = math.ceil(len(trial)/bucket_window)
            this_trial_contours = []
            this_trial_circles = []
            # loop through the trial at given sample rate
            for sample in range(no_of_samples):
                start = sample * bucket_window
                end = (sample * bucket_window) + (bucket_window - 1)
                this_slice = trial[start:end]
                for line in this_slice:
                    if (line<0).any():
                        line[:] = np.nan
                # extract pupil sizes from valid time buckets
                this_slice_contours = []
                this_slice_circles = []
                for frame in this_slice:
                    this_slice_contours.append(frame[2])
                    this_slice_circles.append(frame[5])
                # average the pupil size in this sample slice
                this_slice_avg_contour = np.nanmean(this_slice_contours)            
                this_slice_avg_circle = np.nanmean(this_slice_circles)
                # append to list of downsampled pupil sizes
                this_trial_contours.append(this_slice_avg_contour)
                this_trial_circles.append(this_slice_avg_circle)
            # Find count of bad measurements
            bad_count_contours = sum(np.isnan(this_trial_contours))
            bad_count_circles = sum(np.isnan(this_trial_circles))
            # if more than half of the trial is NaN, then throw away this trial
            # otherwise, if it's a good enough trial...
            if (bad_count_contours < (no_of_samples/2)): 
                this_trial_length = len(this_trial_contours)
                data_contours[index][0:this_trial_length] = this_trial_contours
                data_circles[index][0:this_trial_length] = this_trial_circles
            index = index + 1
        return data_contours, data_circles, num_trials
    else: 
        print("Sample rate must be a multiple of {bucket}".format(bucket=bucket_size))

def list_sub_folders(path_to_root_folder):
    # List all sub folders
    sub_folders = []
    for folder in os.listdir(path_to_root_folder):
        if(os.path.isdir(os.path.join(path_to_root_folder, folder))):
            sub_folders.append(os.path.join(path_to_root_folder, folder))
    return sub_folders

def threshold_to_nan(array_of_arrays, threshold, upper_or_lower):
    for array in array_of_arrays:
        for element in array: 
            if upper_or_lower=='upper':
                if np.isnan(element)==False and element>threshold:
                    element = np.nan
            if upper_or_lower=='lower':
                if np.isnan(element)==False and element<threshold:
                    element = np.nan
    return array_of_arrays

### BEGIN ANALYSIS ###
# grab today's date
now = datetime.datetime.now()
todays_datetime = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')
# List relevant data locations: these are for KAMPFF-LAB-VIDEO
#root_folder = r"C:\Users\KAMPFF-LAB-VIDEO\Dropbox\SurprisingMinds\analysis\pythonWithAdam-csv"
root_folder = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\pythonWithAdam-csv"
current_working_directory = os.getcwd()

# set up log file to store all printed messages
log_filename = "pupil-plotting_log_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
log_file = os.path.join(current_working_directory, log_filename)
sys.stdout = open(log_file, "w")

# set up folders
plots_folder = os.path.join(current_working_directory, "plots")
pupils_folder = os.path.join(plots_folder, "pupils")
engagement_folder = os.path.join(plots_folder, "engagement")

# Create plots folder (and sub-folders) if it (they) does (do) not exist
if not os.path.exists(plots_folder):
    #print("Creating plots folder.")
    os.makedirs(plots_folder)
if not os.path.exists(pupils_folder):
    #print("Creating camera profiles folder.")
    os.makedirs(pupils_folder)
if not os.path.exists(engagement_folder):
    #print("Creating engagement count folder.")
    os.makedirs(engagement_folder)

# consolidate csv files from multiple days into one data structure
day_folders = list_sub_folders(root_folder)

all_right_trials = []
all_left_trials = []
activation_count = []

# downsample = collect data from every 40ms or other multiples of 20
downsample_rate_in_ms = 80
original_bucket_size_in_ms = 4
clip_length_in_time_buckets = 7000

for day_folder in day_folders: 
    # for each day...
    day_folder_path = os.path.join(root_folder, day_folder)
    analysis_folder = os.path.join(day_folder_path, "Analysis")
    csv_folder = os.path.join(analysis_folder, "csv")

    # Print/save number of users per day
    day_name = day_folder.split("_")[-1]
    try: 
        right_area_contours, right_area_circles, num_right_trials = load_daily_pupil_areas("right", csv_folder, clip_length_in_time_buckets, original_bucket_size_in_ms, downsample_rate_in_ms)
        left_area_contours, left_area_circles, num_left_trials = load_daily_pupil_areas("left", csv_folder, clip_length_in_time_buckets, original_bucket_size_in_ms, downsample_rate_in_ms)

        activation_count.append((num_right_trials, num_left_trials))
        print("On {day}, exhibit was activated {count} times".format(day=day_name, count=num_right_trials))

        ## COMBINE EXTRACTING PUPIL SIZE AND POSITION

        # filter data
        # contours that are too big
        right_area_contours = threshold_to_nan(right_area_contours, 15000, 'upper')
        left_area_contours = threshold_to_nan(left_area_contours, 15000, 'upper')
        # time buckets with no corresponding frames
        right_area_contours = threshold_to_nan(right_area_contours, 0, 'lower')
        left_area_contours = threshold_to_nan(left_area_contours, 0, 'lower')

        # create a baseline - take first 3 seconds, aka 75 time buckets where each time bucket is 40ms
        right_area_contours_baseline = np.nanmedian(right_area_contours[:,0:75], 1)
        left_area_contours_baseline = np.nanmedian(left_area_contours[:,0:75], 1)

        # normalize and append
        for index in range(len(right_area_contours_baseline)): 
            right_area_contours[index,:] = right_area_contours[index,:]/right_area_contours_baseline[index]
            all_right_trials.append(right_area_contours[index,:])
        for index in range(len(left_area_contours_baseline)): 
            left_area_contours[index,:] = left_area_contours[index,:]/left_area_contours_baseline[index]
            all_left_trials.append(left_area_contours[index,:])
        print("Day {day} succeeded!".format(day=day_name))
    except Exception:
        print("Day {day} failed!".format(day=day_name))

# Save activation count to csv
engagement_count_filename = 'Exhibit_Activation_Count_measured-' + todays_datetime + '.csv'
csv_file = os.path.join(engagement_folder, engagement_count_filename)
np.savetxt(csv_file, activation_count, fmt='%.2f', delimiter=',')

# Plot activation count
total_activation = sum(count[0] for count in activation_count)
total_days_activated = len(activation_count)
print("Total number of exhibit activations: {total}".format(total=total_activation))
activation_array = np.array(activation_count)

figure_name = 'TotalExhibitActivation_' + todays_datetime + '.pdf'
figure_path = os.path.join(pupils_folder, figure_name)
figure_title = "Total number of exhibit activations per day (Grand Total: " + str(total_activation) + ") \nPlotted on " + todays_datetime
plt.figure(figsize=(7, 6.4), dpi=100)
plt.suptitle(figure_title, fontsize=12, y=0.98)

plt.ylabel('Number of activations', fontsize=11)
plt.xlabel('Days, Total days activated: ' + str(total_days_activated), fontsize=11)
#plt.minorticks_on()
plt.grid(b=True, which='major', linestyle='-')
plt.grid(b=True, which='minor', linestyle='--')
plt.plot(activation_array, color=[0.0, 0.0, 1.0])

plt.savefig(figure_path)
plt.show(block=False)
plt.pause(1)
plt.close()

# turn cleaned-up data into arrays
all_right_trials_array = np.array(all_right_trials)
all_left_trials_array = np.array(all_left_trials)     
# Compute global mean
all_right_contours_mean = np.nanmean(all_right_trials_array, 0)
all_left_contours_mean = np.nanmean(all_left_trials_array, 0)

# Plot pupil sizes
figure_name = 'AveragePupilSizes_' + todays_datetime + '.pdf'
figure_path = os.path.join(pupils_folder, figure_name)
figure_title = "Pupil sizes of participants,  \nPlotted on " + todays_datetime
plt.figure(figsize=(7, 6.4), dpi=100)
plt.suptitle(figure_title, fontsize=12, y=0.98)

plt.subplot(2,1,1)
ax = plt.gca()
ax.yaxis.set_label_coords(-0.09, 0.0) 
plt.ylabel('Percentage from baseline', fontsize=11)
plt.title('Right eye pupil sizes', fontsize=9, color='grey', style='italic')
plt.minorticks_on()
plt.grid(b=True, which='major', linestyle='-')
plt.grid(b=True, which='minor', linestyle='--')
plt.plot(all_right_trials_array.T, '.', MarkerSize=1, color=[0.0, 0.0, 1.0, 0.01])
plt.plot(all_right_contours_mean, linewidth=4, color=[1.0, 0.0, 0.0, 0.3])
plt.ylim(0,2)

plt.subplot(2,1,2)
plt.xlabel('Time buckets (downsampled, 1 time bucket = ' + str(downsample_rate_in_ms) + 'ms)', fontsize=11)
plt.title('Left eye pupil sizes', fontsize=9, color='grey', style='italic')
plt.minorticks_on()
plt.grid(b=True, which='major', linestyle='-')
plt.grid(b=True, which='minor', linestyle='--')
plt.plot(all_left_trials_array.T, '.', MarkerSize=1, color=[0.0, 1.0, 0.0, 0.01])
plt.plot(all_left_contours_mean, linewidth=4, color=[1.0, 0.0, 0.0, 0.3])
plt.ylim(0,2)

plt.savefig(figure_path)
plt.show()




#FIN