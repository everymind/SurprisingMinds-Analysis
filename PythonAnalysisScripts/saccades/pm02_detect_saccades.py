##################################################################
##################################################################
# Load speed files and SEPARATE INTO CALIB, OCTO, AND UNIQUE STIM
# save intermediate files that are calib, octo, and unique
##################################################################
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os.path

# grab today's date
now = datetime.datetime.now()
todays_datetime = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')
current_working_directory = os.getcwd()

# Specify relevant data/output folders - laptop
data_folder = r'C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\intermediates'
plots_folder = r'C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\plots\saccade_detector'
# Specify relevant data/output folders - office
#data_folder = r'C:\Users\Kampff_Lab\Dropbox\SurprisingMinds\analysis\dataPythonWorkflows'
#plots_folder = r'C:\Users\Kampff_Lab\Dropbox\SurprisingMinds\analysis\plots\saccade_detector'

# make separate folder for each sequence
# [CAUTION, DELETES ALL PREVIOUS SPEED FILES]
calib_folder = data_folder + os.sep + 'calib_peaks'
octo_folder = data_folder + os.sep + 'octo_peaks'
sequences = [calib_folder, octo_folder]
unique_folders = {}
for stim in range(6):
    this_stim_folder = data_folder + os.sep + 'stim' + str(stim) + '_peaks'
    sequences.append(this_stim_folder)
    unique_folders[stim] = this_stim_folder
for seq_folder in sequences:
    if not os.path.exists(seq_folder):
        #print("Creating plots folder.")
        os.makedirs(seq_folder)
    if os.path.exists(seq_folder):
        # make sure it's empty
        filelist = glob.glob(os.path.join(seq_folder, "*.npz"))
        for f in filelist:
            os.remove(f)

# load speed files
speed_data_folder = data_folder + os.sep + 'speeds'
trial_len_cutoff = 20000
speed_files = glob.glob(speed_data_folder + os.sep + '*.data')
num_files = len(speed_files)
#peak_raster = np.zeros((num_files, trial_len_cutoff))
#speed_raster = np.zeros((num_files, trial_len_cutoff))

#window_size = 50
#all_peak_windows = np.zeros((0,window_size))
#all_peak_speeds = np.zeros(0)
#all_peak_durations = np.zeros(0)
#all_peak_intervals = np.zeros(0)

# set time points for sequences (resolution: 4ms timebuckets)
calib_start = 0
calib_end = 4431
unique_start = 4431
unique_ends = {0: 5962, 1: 6020, 2: 6660, 3: 6080, 4: 6670, 5: 7190}
octo_len = 3980

# initiate trial counters for each sequence
calib_trials = 0
octo_trials = 0
unique_trials = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}

# cut speed data into sequences
for s in range(6):
    count = 0
    for i, speed_file in enumerate(speed_files):
        # Get stimulus number
        trial_name = os.path.basename(speed_file)
        fields = trial_name.split(sep='_')
        eye = fields[1]
        stimulus = int(fields[0][-1])
        # Check if current stimulus number
        if(stimulus == s):
            # Load speed_file
            speed = np.fromfile(speed_file, dtype=np.float32)
            if len(speed) < trial_len_cutoff:
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
                # for peak_index in peak_indices:
                #     left_border = np.int(peak_index - np.round(window_size/2))
                #     right_border = np.int(left_border + window_size)

                #     # Check that window fits
                #     if left_border < 0:
                #         continue
                #     if right_border > len(speed):
                #         continue
                
                # categorise peaks according to the sequence they happened within
                # peak speeds
                calib_peaks_speeds = []
                octo_peaks_speeds = []
                unique_peaks_speeds = []
                # peak indices
                calib_peaks_indices = []
                octo_peaks_indices = []
                unique_peaks_indices = []
                for i, peak in enumerate(peak_indices):
                    if peak<=calib_end:
                        calib_peaks_indices.append(peak)
                        calib_peaks_speeds.append(peak_speeds[i])
                    if unique_start<peak<=unique_ends[stimulus]:
                        unique_peaks_indices.append(peak)
                        unique_peaks_speeds.append(peak_speeds[i])
                    if unique_ends[stimulus]<peak<(unique_ends[stimulus]+octo_len):
                        octo_peaks_indices.append(peak)
                        octo_peaks_speeds.append(peak_speeds[i])
                calib_peaks_speeds = np.array(calib_peaks_speeds)
                octo_peaks_speeds = np.array(octo_peaks_speeds)
                unique_peaks_speeds = np.array(unique_peaks_speeds)
                calib_peaks_indices = np.array(calib_peaks_indices)
                octo_peaks_indices = np.array(octo_peaks_indices)
                unique_peaks_indices = np.array(unique_peaks_indices)
                # Store
                # calibration
                calib_path = calib_folder + os.sep + 'stim%d_%s_calib-peaks_%d.npz' % (stimulus, eye, calib_trials)
                np.savez(calib_path, speeds=calib_peaks_speeds, indices=calib_peaks_indices)
                calib_trials = calib_trials + 1
                # octo
                octo_path = octo_folder + os.sep + 'stim%d_%s_octo-peaks_%d.npz' % (stimulus, eye, octo_trials)
                np.savez(octo_path, speeds=octo_peaks_speeds, indices=octo_peaks_indices)
                octo_trials = octo_trials + 1
                # unique
                unique_path = unique_folders[stimulus] + os.sep + 'stim%d_%s_unique-peaks_%d.npz' % (stimulus, eye, unique_trials[stimulus])
                np.savez(unique_path, speeds=unique_peaks_speeds, indices=unique_peaks_indices)
                unique_trials[stimulus] = unique_trials[stimulus] + 1
                # report progress
                print('Calib trial count: {c}'.format(c=calib_trials))
                print('Octo trial count: {o}'.format(o=octo_trials))
                print('Unique stim {s} count: {u}'.format(s=stimulus, u=unique_trials[stimulus]))
                print('---')
                print('---')


# FIN
