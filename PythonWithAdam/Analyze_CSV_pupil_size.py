import os
import glob
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

### FUNCTIONS ###
def load_daily_pupil_areas(which_eye, day_folder_path, max_no_of_buckets, original_bucket_size, new_bucket_size): 
    if (new_bucket_size % original_bucket_size == 0):
        new_sample_rate = int(new_bucket_size/original_bucket_size)
        print("New bucket window = {size}, need to average every {sample_rate} buckets".format(size=new_bucket_size, sample_rate=new_sample_rate))
        downsampled_no_of_buckets = math.ceil(max_no_of_buckets/new_sample_rate)
        print("Downsampled number of buckets = {number}".format(number=downsampled_no_of_buckets))
        # List all csv trial files
        trial_files = glob.glob(day_folder_path + os.sep + which_eye + "*.csv")
        num_trials = len(trial_files)
        # contours
        data_contours = np.empty((num_trials, downsampled_no_of_buckets))
        data_contours[:] = -6
        # circles
        data_circles = np.empty((num_trials, downsampled_no_of_buckets))
        data_circles[:] = -6

        index = 0
        for trial_file in trial_files:
            trial_name = trial_file.split(os.sep)[-1]
            trial = np.genfromtxt(trial_file, dtype=np.float, delimiter=",")
            no_of_samples = math.ceil(len(trial)/new_sample_rate)
            this_trial_contours = []
            this_trial_circles = []
            # loop through the trial at given sample rate
            for sample in range(no_of_samples):
                start = sample * new_sample_rate
                end = (sample * new_sample_rate) + (new_sample_rate - 1)
                this_slice = trial[start:end]
                for line in this_slice:
                    if (line<0).any():
                        line[:] = np.nan
                    if (line>15000).any():
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
                this_chunk_length = len(this_trial_contours)
                data_contours[index][0:this_chunk_length] = this_trial_contours
                data_circles[index][0:this_chunk_length] = this_trial_circles
            index = index + 1
        return data_contours, data_circles, num_trials
    else: 
        print("Sample rate must be a multiple of {bucket}".format(bucket=original_bucket_size))

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
plots_folder = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\plots"
pupils_folder = os.path.join(plots_folder, "pupil")
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
# first day was a debugging session, so skip it
day_folders = day_folders[1:]
# currently still running pupil finding analysis...
day_folders = day_folders[:-1]

all_right_trials_contours = []
all_right_trials_circles = []
all_left_trials_contours = []
all_left_trials_circles = []
activation_count = []

# downsample = collect data from every 40ms or other multiples of 20
downsample_rate_ms = 60
original_bucket_size_in_ms = 4
no_of_time_buckets = 10000
new_time_bucket_ms = downsample_rate_ms/original_bucket_size_in_ms

for day_folder in day_folders: 
    # for each day...
    day_folder_path = os.path.join(root_folder, day_folder)
    analysis_folder = os.path.join(day_folder_path, "Analysis")
    csv_folder = os.path.join(analysis_folder, "csv")

    # Print/save number of users per day
    day_name = day_folder.split("_")[-1]
    try: 
        right_area_contours, right_area_circles, num_right_trials = load_daily_pupil_areas("right", csv_folder, no_of_time_buckets, original_bucket_size_in_ms, downsample_rate_ms)
        left_area_contours, left_area_circles, num_left_trials = load_daily_pupil_areas("left", csv_folder, no_of_time_buckets, original_bucket_size_in_ms, downsample_rate_ms)

        activation_count.append((num_right_trials, num_left_trials))
        print("On {day}, exhibit was activated {count} times".format(day=day_name, count=num_right_trials))

        ## COMBINE EXTRACTING PUPIL SIZE AND POSITION

        # filter data
        # contours/circles that are too big
        right_area_contours = threshold_to_nan(right_area_contours, 15000, 'upper')
        right_area_circles = threshold_to_nan(right_area_circles, 15000, 'upper')
        left_area_contours = threshold_to_nan(left_area_contours, 15000, 'upper')
        left_area_circles = threshold_to_nan(left_area_circles, 15000, 'upper')
        # time buckets with no corresponding frames
        right_area_contours = threshold_to_nan(right_area_contours, 0, 'lower')
        right_area_circles = threshold_to_nan(right_area_circles, 0, 'lower')
        left_area_contours = threshold_to_nan(left_area_contours, 0, 'lower')
        left_area_circles = threshold_to_nan(left_area_circles, 0, 'lower')

        # create a baseline - take first 3 seconds, aka 75 time buckets where each time bucket is 40ms
        milliseconds_for_baseline = 3000
        baseline_no_buckets = int(milliseconds_for_baseline/new_time_bucket_ms)
        right_area_contours_baseline = np.nanmedian(right_area_contours[:,0:baseline_no_buckets], 1)
        right_area_circles_baseline = np.nanmedian(right_area_circles[:,0:baseline_no_buckets], 1)
        left_area_contours_baseline = np.nanmedian(left_area_contours[:,0:baseline_no_buckets], 1)
        left_area_circles_baseline = np.nanmedian(left_area_circles[:,0:baseline_no_buckets], 1)

        # normalize and append
        for index in range(len(right_area_contours_baseline)): 
            right_area_contours[index,:] = right_area_contours[index,:]/right_area_contours_baseline[index]
            all_right_trials_contours.append(right_area_contours[index,:])
        for index in range(len(right_area_circles_baseline)): 
            right_area_circles[index,:] = right_area_circles[index,:]/right_area_circles_baseline[index]
            all_right_trials_circles.append(right_area_circles[index,:])
        for index in range(len(left_area_contours_baseline)): 
            left_area_contours[index,:] = left_area_contours[index,:]/left_area_contours_baseline[index]
            all_left_trials_contours.append(left_area_contours[index,:])
        for index in range(len(left_area_circles_baseline)): 
            left_area_circles[index,:] = left_area_circles[index,:]/left_area_circles_baseline[index]
            all_left_trials_circles.append(left_area_circles[index,:])
        print("Day {day} succeeded!".format(day=day_name))
    except Exception:
        print("Day {day} failed!".format(day=day_name))

### EXHIBIT ACTIVITY METADATA ### 
# Save activation count to csv
engagement_count_filename = 'Exhibit_Activation_Count_measured-' + todays_datetime + '.csv'
engagement_data_folder = os.path.join(current_working_directory, 'Exhibit-Engagement')
if not os.path.exists(engagement_data_folder):
    #print("Creating plots folder.")
    os.makedirs(engagement_data_folder)
csv_file = os.path.join(engagement_data_folder, engagement_count_filename)
np.savetxt(csv_file, activation_count, fmt='%.2f', delimiter=',')

# Plot activation count
total_activation = sum(count[0] for count in activation_count)
total_days_activated = len(activation_count)
print("Total number of exhibit activations: {total}".format(total=total_activation))
activation_array = np.array(activation_count)
# do da plot
image_type_options = ['.png', '.pdf']
for image_type in image_type_options:
    figure_name = 'TotalExhibitActivation_' + todays_datetime + image_type
    figure_path = os.path.join(engagement_folder, figure_name)
    figure_title = "Total number of exhibit activations per day (Grand Total: " + str(total_activation) + ") \nPlotted on " + todays_datetime
    plt.figure(figsize=(12, 6), dpi=200)
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

### BACK TO THE PUPILS ###
all_right_trials_contours_array = np.array(all_right_trials_contours)
all_right_trials_circles_array = np.array(all_right_trials_circles)
all_left_trials_contours_array = np.array(all_left_trials_contours)
all_left_trials_circles_array = np.array(all_left_trials_circles)
trials_to_plot = [(all_right_trials_contours_array, all_left_trials_contours_array), (all_right_trials_circles_array, all_left_trials_circles_array)]

# Compute global mean
all_right_contours_mean = np.nanmean(all_right_trials_contours_array, 0)
all_right_circles_mean = np.nanmean(all_right_trials_circles_array, 0)
all_left_contours_mean = np.nanmean(all_left_trials_contours_array, 0)
all_left_circles_mean = np.nanmean(all_left_trials_circles_array, 0)
means_to_plot = [(all_right_contours_mean, all_left_contours_mean), (all_right_circles_mean, all_left_circles_mean)]

# event locations in time
milliseconds_until_octo_fully_visible = 6575
milliseconds_until_octopus_inks = 11500
milliseconds_until_octopus_disappears = 11675
milliseconds_until_camera_out_of_ink_cloud = 13000
milliseconds_until_thankyou_screen = 15225
event_labels = ['Octopus video clip starts', 'Octopus fully visible', 'Octopus begins inking', 'Octopus disappears from camera view', 'Camera exits ink cloud', 'Thank you screen']
tb_octo_visible = milliseconds_until_octo_fully_visible/downsample_rate_ms
tb_octo_inks = milliseconds_until_octopus_inks/downsample_rate_ms
tb_octo_disappears = milliseconds_until_octopus_disappears/downsample_rate_ms
tb_camera_out_of_ink_cloud = milliseconds_until_camera_out_of_ink_cloud/downsample_rate_ms
tb_thankyou_screen = milliseconds_until_thankyou_screen/downsample_rate_ms
event_locations = np.array([0, tb_octo_visible, tb_octo_inks, tb_octo_disappears, tb_camera_out_of_ink_cloud, tb_thankyou_screen])

plot_types = ["contours", "circles"]
# Plot pupil sizes
for i in range(len(trials_to_plot)):
    plot_type_right = trials_to_plot[i][0]
    plot_type_left = trials_to_plot[i][1]
    plot_means_right = means_to_plot[i][0]
    plot_means_left = means_to_plot[i][1]
    plot_type_name = plot_types[i]
    for image_type in image_type_options:
        if (image_type == '.pdf'):
            continue
        else:
            dpi_sizes = [400]
        for size in dpi_sizes: 
            figure_name = 'AveragePupilSizes_' + plot_type_name + '_' + todays_datetime + '_dpi' + str(size) + image_type 
            figure_path = os.path.join(pupils_folder, figure_name)
            figure_title = "Pupil sizes of participants, N=" + str(total_activation) +"\nAnalysis type: " + plot_type_name + "\nPlotted on " + todays_datetime
            plt.figure(figsize=(12, 7), dpi=size)
            plt.suptitle(figure_title, fontsize=12, y=0.98)

            plt.subplot(2,1,1)
            ax = plt.gca()
            ax.yaxis.set_label_coords(-0.09, 0.0) 
            plt.ylabel('Percentage from baseline', fontsize=11)
            plt.title('Right eye pupil sizes', fontsize=9, color='grey', style='italic')
            plt.minorticks_on()
            plt.grid(b=True, which='major', linestyle='-')
            plt.grid(b=True, which='minor', linestyle='--')
            plt.plot(plot_type_right.T, '.', MarkerSize=1, color=[0.0, 0.0, 1.0, 0.01])
            plt.plot(plot_means_right, linewidth=1.5, color=[1.0, 0.0, 0.0, 0.4])
            plt.xlim(-10,330)
            plt.ylim(0,2.5)
            # mark events
            for i in range(len(event_labels)):
                plt.plot((event_locations[i],event_locations[i]), (0.25,2.2-((i-1)/5)), 'k-', linewidth=1)
                plt.text(event_locations[i]+1,2.2-((i-1)/5), event_labels[i], fontsize='x-small', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.35'))
            
            plt.subplot(2,1,2)
            plt.xlabel('Time buckets (downsampled, 1 time bucket = ' + str(downsample_rate_ms) + 'ms)', fontsize=11)
            plt.title('Left eye pupil sizes', fontsize=9, color='grey', style='italic')
            plt.minorticks_on()
            plt.grid(b=True, which='major', linestyle='-')
            plt.grid(b=True, which='minor', linestyle='--')
            plt.plot(plot_type_left.T, '.', MarkerSize=1, color=[0.0, 1.0, 0.0, 0.01])
            plt.plot(plot_means_left, linewidth=1.5, color=[1.0, 0.0, 0.0, 0.4])
            plt.xlim(-10,330)
            plt.ylim(0,2.5)
            # mark events
            for i in range(len(event_labels)):
                plt.plot((event_locations[i],event_locations[i]), (0.25,2.2-((i-1)/5)), 'k-', linewidth=1)
                plt.text(event_locations[i]+1,2.2-((i-1)/5), event_labels[i], fontsize='x-small', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.35'))
            

            plt.savefig(figure_path)
            plt.show(block=False)
            plt.pause(1)
            plt.close()


#FIN