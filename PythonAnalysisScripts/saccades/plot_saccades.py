import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os.path

# Plot saccade figures

# Grab today's date
now = datetime.datetime.now()
todays_datetime = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')
current_working_directory = os.getcwd()

# Specify relevant data/output folders - laptop
data_folder = r'C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\dataPythonWorkflows\speeds'
plots_folder = r'C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\plots\saccade_detector'
folder_name = 'SurprisingMinds*'
# Specify relevant data/output folders - office (windows)
#data_folder = r'C:\Users\Kampff_Lab\Dropbox\SurprisingMinds\analysis\pythonWithAdam-csv\speeds'
#plots_folder = r'C:\Users\Kampff_Lab\Dropbox\SurprisingMinds\analysis\plots\saccade_detector'
#folder_name = 'SurprisingMinds*'
# Specify relevant data/output folders - office (linux)
#data_folder = r'/home/kampff/Data/Surprising/saccades'
#plots_folder = r'/home/kampff/Data/Surprising/figures'
#folder_name = 'surprisingminds*'

# Create a folder for figures (if it does not exist)
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)

######################
# Load saccade files
######################
trial_len_cutoff = 20000
saccade_files = glob.glob(data_folder + os.sep + '*.data')
num_files = len(saccade_files)
window_size = 50

# Plot saccades in each stimuli category
trial_count = 0
plt.figure()
for i, saccade_file in enumerate(saccade_files):

    # Get stimulus number and eye
    trial_name = os.path.basename(saccade_file)
    fields = trial_name.split(sep='_')
    eye = fields[1]
    stimulus = int(fields[0][-1])

    # Load saccades for this trial
    flat_saccades = np.fromfile(saccade_file, dtype=np.float32)
    saccades = np.reshape(flat_saccades, (-1, 4 + window_size))

    # Make lame cluster plot
    plt.plot(saccades[:,0], saccades[:,2], 'k.', alpha=0.005)

    # Seperate data structures for stim type, (calib phase saccades, stimulus phase saccades, octupus saccades)    

# Display
plt.show()

#FIN