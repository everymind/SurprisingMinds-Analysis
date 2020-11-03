##########################################
### PLOT SACCADE RASTER for each sequence
##########################################
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

# set boundaries for categories of saccade
big_upper = 75
big_lower = 45
med_upper = 25
med_lower = 15
lil_upper = 15
lil_lower = 1

fsize = 200 #dpi

# collect peak files for each sequence
seq_peak_files = {}
seq_peak_files['calib'] = glob.glob(data_folder + os.sep + '*calib_peaks' + os.sep + '*.npz')
seq_peak_files['octo'] = glob.glob(data_folder + os.sep + '*octo_peaks' + os.sep + '*.npz')
for stim in range(6):
    this_unique_peak_files = glob.glob(data_folder + os.sep + 'stim' + str(stim) + '_peaks' + os.sep + '*.npz')
    seq_peak_files[str(stim)] = this_unique_peak_files

# plot rasters of saccades during each sequence
for seq_type in seq_peak_files.keys():
    peak_files = seq_peak_files[seq_type]
    seq_trial_count = len(peak_files)
    # set figure save path and title
    figure_name = 'DetectedSaccades_' + seq_type + '_' + todays_datetime + '.png'
    figure_path = os.path.join(plots_folder, figure_name)
    figure_title = 'Detected Saccades during sequence {s}, categorized by speed, N={n}'.format(s=seq_type, n=seq_trial_count)
    plt.figure(figsize=(14, 14), dpi=fsize)
    plt.suptitle(figure_title, fontsize=12, y=0.98)
    count = 0
    for i, peak_file in enumerate(peak_files):
        # Get stimulus number
        trial_name = os.path.basename(peak_file)
        fields = trial_name.split(sep='_')
        eye = fields[1]
        stimulus = int(fields[0][-1])
        seq = fields[2].split('-')[0]
        # Load peak_file
        peaks = np.load(peak_file)
        peak_speeds = peaks['speeds']
        peak_indices = peaks['indices']
        # Make some peak categories
        big_speeds = (peak_speeds < big_upper) * (peak_speeds > big_lower)
        med_speeds = (peak_speeds < med_upper) * (peak_speeds > med_lower)
        lil_speeds = (peak_speeds < lil_upper) * (peak_speeds > lil_lower)
        # Plot a saccade raster
        num_peaks = np.sum(big_speeds)
        row_value = count*np.ones(num_peaks)
        plt.subplot(3,1,1)
        plt.ylabel('Individual Trials', fontsize=9)
        plt.title('Big Saccades (pupil movements between {l} and {u} pixels per frame)'.format(l=big_lower, u=big_upper), fontsize=10, color='grey', style='italic')
        plt.plot(peak_indices[big_speeds], row_value, 'r.', alpha=0.05)

        num_peaks = np.sum(med_speeds)
        row_value = count*np.ones(num_peaks)
        plt.subplot(3,1,2)
        plt.ylabel('Individual Trials', fontsize=9)
        plt.title('Medium Saccades (pupil movements between {l} and {u} pixels per frame)'.format(l=med_lower, u=med_upper), fontsize=10, color='grey', style='italic')
        plt.plot(peak_indices[med_speeds], row_value, 'b.', alpha=0.05)

        num_peaks = np.sum(lil_speeds)
        row_value = count*np.ones(num_peaks)
        plt.subplot(3,1,3)
        plt.ylabel('Individual Trials', fontsize=9)
        plt.title('Small Saccades (pupil movements between {l} and {u} pixels per frame)'.format(l=lil_lower, u=lil_upper), fontsize=10, color='grey', style='italic')
        plt.plot(peak_indices[lil_speeds], row_value, 'k.', alpha=0.1)

        # Add to all "peak" arrays
        #all_peak_speeds = np.hstack((all_peak_speeds, peak_speeds))
        #all_peak_durations = np.hstack((all_peak_durations, peak_durations))
        #all_peak_intervals = np.hstack((all_peak_intervals, peak_intervals))

        # Store peaks in peak raster
        #peak_raster[count, peak_intervals] = 1

        # Report
        print(count)
        print("--")
        print("--")
        count = count + 1
    # save and display
    #plt.subplots_adjust(hspace=0.5)
    plt.savefig(figure_path)
    plt.show(block=False)
    plt.pause(1)
    plt.close()


# FIN