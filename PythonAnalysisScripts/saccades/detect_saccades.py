import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os.path

# Detectes saccades in each speed file and saves a new intermediate binary file

# Grab today's date
now = datetime.datetime.now()
todays_datetime = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')
current_working_directory = os.getcwd()

# Specify relevant data/output folders - laptop
#data_folder = r'C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\dataPythonWorkflows'
#folder_name = 'SurprisingMinds*'
# Specify relevant data/output folders - office (windows)
data_folder = r'C:\Users\Kampff_Lab\Dropbox\SurprisingMinds\analysis\dataPythonWorkflows'
folder_name = 'SurprisingMinds*'
# Specify relevant data/output folders - office (linux)
#data_folder = r'/home/kampff/Data/Surprising'
#folder_name = 'surprisingminds*'

# Create an empty folder for saccade data [CAUTION, DELETES ALL PREVIOUS SACCADE FILES]
saccade_data_folder = data_folder + os.sep + 'saccades'
if not os.path.exists(saccade_data_folder):
    os.makedirs(saccade_data_folder)
if os.path.exists(saccade_data_folder):
    # make sure it's empty
    filelist = glob.glob(os.path.join(saccade_data_folder, "*.data"))
    for f in filelist:
        os.remove(f)

######################
# Load speed files
######################
speeds_data_folder = data_folder + os.sep + 'speeds'
trial_len_cutoff = 20000
speed_files = glob.glob(speeds_data_folder + os.sep + '*.data')
#speed_files = speed_files[0:10]
num_files = len(speed_files)
window_size = 50

# Detect (and filter) saccades in each stimuli category
trial_count = 0
for i, speed_file in enumerate(speed_files):

    # Get stimulus number and eye
    trial_name = os.path.basename(speed_file)
    fields = trial_name.split(sep='_')
    eye = fields[1]
    stimulus = int(fields[0][-1])

    # Load speed_file
    speed = np.fromfile(speed_file, dtype=np.float32)
    if len(speed) >= trial_len_cutoff:
        continue
    else:
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
        #good_peaks = (peak_intervals > 25) * (peak_durations < 30) * (peak_durations > 4) * (peak_speeds < 100)
        good_peaks = (peak_intervals > 25)
        good_peaks *= (peak_durations < 30)
        good_peaks *= (peak_durations > 4) 
        good_peaks *= (peak_speeds < 75)
        good_peaks *= (peak_indices > window_size) 
        good_peaks *= (peak_indices < (len(speed) - window_size))

        # Apply filter        
        peak_speeds = peak_speeds[good_peaks]
        peak_indices = peak_indices[good_peaks]
        peak_durations = peak_durations[good_peaks]
        peak_intervals = peak_intervals[good_peaks]

        # Save all (good)peaks (max_speed, index, duration, inter-peak-interval, and window) to a saccade file
        num_saccades = np.sum(good_peaks)

        # Allocate space for saccade 2-d array
        saccades = np.zeros((num_saccades, 4 + window_size), dtype=np.float32)

        # Extract windows around peak maxima
        for s in range(num_saccades):
            peak_index = peak_indices[s]
            left_border = np.int(peak_index - np.round(window_size/2))
            right_border = np.int(left_border + window_size)
            peak_window = speed[left_border:right_border]

            # Fill sacaades 2D array
            saccades[s, 0] = peak_speeds[s]
            saccades[s, 1] = peak_indices[s]
            saccades[s, 2] = peak_durations[s]
            saccades[s, 3] = peak_intervals[s]
            saccades[s, 4:] = peak_window

        # Store saccade file
        output_path = saccade_data_folder + os.sep + 'stim%d_%s_saccade_%d.data' % (stimulus, eye, trial_count)
        saccades.tofile(output_path)

        # Report and increment
        print(trial_count)
        trial_count = trial_count + 1

#FIN