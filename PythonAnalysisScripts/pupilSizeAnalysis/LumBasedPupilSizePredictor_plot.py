### --------------------------------------------------------------------------- ###
# loads normalised mean pupil sizes for each stimulus phase
# generates prediction of pupil size based on avg lum of stimuli
# compares to observed pupil sizes to find moments where observed deviates from predicted pupil size
### --------------------------------------------------------------------------- ###
import os
import numpy as np 
import matplotlib.pyplot as plt
import glob

###################################
# FUNCTIONS
###################################

###################################
# DATA AND OUTPUT FILE LOCATIONS
###################################
# List relevant data locations: these are for laptop
root_folder = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\dataPythonWorkflows"
plots_folder = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\plots"
# List relevant data locations: these are for office desktop
#root_folder = r"C:\Users\Kampff_Lab\Dropbox\SurprisingMinds\analysis\dataPythonWorkflows"
#plots_folder = r"C:\Users\Kampff_Lab\Dropbox\SurprisingMinds\analysis\plots"
# pupil size data
normedMeanPupils_folder = os.path.join(root_folder, 'normedMeanPupilSizes')
# linear regression data
linRegressParams_folder = os.path.join(root_folder, 'pupilSizeVsDelayLinRegress')

###################################
# TIMING/SAMPLING VARIABLES FOR DATA EXTRACTION
###################################
# downsample = collect data from every 40ms or other multiples of 20
downsampled_bucket_size_ms = 40
original_bucket_size_in_ms = 4

###################################
# load linear regression parameters for pupil size vs avg luminance of stimulus video
# organise by eyeAnalysis_type, phase_type, and delay_ms
###################################
# extract linear regression parameters to find best delay
linRegressParams_files = glob.glob(linRegressParams_folder + os.sep + '*.npy')
for linReg_file in linRegressParams_files:
    eyeAnalysis = os.path.basename(linReg_file).split('.')[0].split('_')[-1]
    phaseOrAll = os.path.basename(linReg_file).split('.')[0].split('_')[1]
    if phaseOrAll == 'allPhasesConcat':
        thisEyeAnalysis_linRegParams = np.load(linReg_file)
        best_rVal = 0
        best_delay = 0
        for i, delay in enumerate(thisEyeAnalysis_linRegParams):
            if delay[2]<best_rVal:
                best_rVal = delay[2]
                best_delay = i
        if eyeAnalysis == 'RightContours':
            Rco_linRegParams = thisEyeAnalysis_linRegParams
            Rco_bestRVal = best_rVal
            Rco_bestDelay_index = best_delay
        if eyeAnalysis == 'RightCircles':
            Rci_linRegParams = thisEyeAnalysis_linRegParams
            Rci_bestRVal = best_rVal
            Rci_bestDelay_index = best_delay
        if eyeAnalysis == 'LeftContours':
            Lco_linRegParams = thisEyeAnalysis_linRegParams
            Lco_bestRVal = best_rVal
            Lco_bestDelay_index = best_delay
        if eyeAnalysis == 'LeftCircles':
            Lci_linRegParams = thisEyeAnalysis_linRegParams
            Lci_bestRVal = best_rVal
            Lci_bestDelay_index = best_delay
# now we should have:
# Rco_linRegParams, Rco_bestRVal, Rco_bestDelay_index
# Rci_linRegParams, Rci_bestRVal, Rci_bestDelay_index
# Lco_linRegParams, Lco_bestRVal, Lco_bestDelay_index
# Lci_linRegParams, Lci_bestRVal, Lci_bestDelay_index
Rco_bestDelay_ms = Rco_bestDelay_index*downsampled_bucket_size_ms
Rci_bestDelay_ms = Rci_bestDelay_index*downsampled_bucket_size_ms
Lco_bestDelay_ms = Lco_bestDelay_index*downsampled_bucket_size_ms
Lci_bestDelay_ms = Lci_bestDelay_index*downsampled_bucket_size_ms
###################################
# load normalised mean pupil size files
# organise by eyeAnalysis_type, phase_type, and delay_ms
###################################
# extract mean pupil sizes for best delay
normedMeanPupils_files = glob.glob(normedMeanPupils_folder + os.sep + '*.npy')
Rco_bestDelay_allPhases = []
Rco_bestDelay_phaseOrder = []
Rci_bestDelay_allPhases = []
Rci_bestDelay_phaseOrder = []
Lco_bestDelay_allPhases = []
Lco_bestDelay_phaseOrder = []
Lci_bestDelay_allPhases = []
Lci_bestDelay_phaseOrder = []
for nmpSize_file in normedMeanPupils_files:
    delay_ms = int(os.path.basename(nmpSize_file).split('_')[2][:-7])
    eyeAnalysis_type = os.path.basename(nmpSize_file).split('_')[3][:-4]
    phase_type = os.path.basename(nmpSize_file).split('_')[1]
    if eyeAnalysis_type == 'RightContours':
        if delay_ms == Rco_bestDelay_ms:
            this_nmpSize = np.load(nmpSize_file)
            Rco_bestDelay_allPhases.append(this_nmpSize)
            Rco_bestDelay_phaseOrder.append(phase_type)
    if eyeAnalysis_type == 'RightCircles':
        if delay_ms == Rci_bestDelay_ms:
            this_nmpSize = np.load(nmpSize_file)
            Rci_bestDelay_allPhases.append(this_nmpSize)
            Rci_bestDelay_phaseOrder.append(phase_type)
    if eyeAnalysis_type == 'LeftContours':
        if delay_ms == Lco_bestDelay_ms:
            this_nmpSize = np.load(nmpSize_file)
            Lco_bestDelay_allPhases.append(this_nmpSize)
            Lco_bestDelay_phaseOrder.append(phase_type)
    if eyeAnalysis_type == 'LeftCircles':
        if delay_ms == Lci_bestDelay_ms:
            this_nmpSize = np.load(nmpSize_file)
            Lci_bestDelay_allPhases.append(this_nmpSize)
            Lci_bestDelay_phaseOrder.append(phase_type)



# FIN