### --------------------------------------------------------------------------- ###
# loads monthly mean raw live stim and world cam luminance data files (saved at 4ms resolution)
# saves sanity check mean world cam video for each stimulus type
# measures display latency (lag between when bonsai tells a frame to display and when it actually displays)
# saves binary files of monthly mean raw live stim data with display latency
# NOTE: this script uses ImageMagick to easily install ffmpeg onto Windows 10 (https://www.imagemagick.org/script/download.php)
# NOTE: in command line run with optional tags 
#       1) '--a debug' to use only a subset of pupil location/size data
#       2) '--loc *' to run with various root data locations (see first function below)
### --------------------------------------------------------------------------- ###
import logging
import pdb
import os
import glob
import datetime
import math
import sys
import itertools
import csv
import fnmatch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import stats
import argparse
###################################
# SET CURRENT WORKING DIRECTORY
###################################
current_working_directory = os.getcwd()
###################################
# SCRIPT LOGGER
###################################
# grab today's date
now = datetime.datetime.now()
logging.basicConfig(filename="psa02_DisplayLatency_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".log", filemode='w', level=logging.DEBUG)
###################################
# FUNCTIONS
###################################

##########################################################
#### MODIFY THIS FIRST FUNCTION BASED ON THE LOCATIONS OF:
# 1) root_folder (parent folder with all intermediate data)
# AND
# 2) plots_folder (parent folder for all plots output from analysis scripts)
### Current default usess a debugging source dataset
##########################################################
def load_data(location='laptop'):
    if location == 'laptop':
        root_folder = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\dataPythonWorkflows"
        plots_folder = r"C:\Users\taunsquared\Dropbox\SurprisingMinds\analysis\plots"
    elif location == 'office':
        root_folder = r"C:\Users\Kampff_Lab\Dropbox\SurprisingMinds\analysis\dataPythonWorkflows"
        plots_folder = r"C:\Users\Kampff_Lab\Dropbox\SurprisingMinds\analysis\plots"
    # monthly mean raw live and world cam luminances
    monthly_mean_lums_folders = fnmatch.filter(sorted(os.listdir(root_folder)), 'MeanStimuli_*')
    # plots output folder
    pupilSize_folder = os.path.join(plots_folder, "pupilSizeAnalysis")
    Rco_scatter_folder = os.path.join(pupilSize_folder, 'rightContours', 'scatter')
    Rci_scatter_folder = os.path.join(pupilSize_folder, 'rightCircles', 'scatter')
    Lco_scatter_folder = os.path.join(pupilSize_folder, 'leftContours', 'scatter')
    Lci_scatter_folder = os.path.join(pupilSize_folder, 'leftCircles', 'scatter')
    Rco_rvalVsDelay_folder = os.path.join(pupilSize_folder, 'rightContours', 'rvalVsDelay')
    Rci_rvalVsDelay_folder = os.path.join(pupilSize_folder, 'rightCircles', 'rvalVsDelay')
    Lco_rvalVsDelay_folder = os.path.join(pupilSize_folder, 'leftContours', 'rvalVsDelay')
    Lci_rvalVsDelay_folder = os.path.join(pupilSize_folder, 'leftCircles', 'rvalVsDelay')
    # normed mean pupil sizes output folder
    normedMeanPupilSizes_folder = os.path.join(root_folder, 'normedMeanPupilSizes')
    pupilSizeVsDelayLinRegress_folder = os.path.join(root_folder, 'pupilSizeVsDelayLinRegress')
    # Create output folders if they do not exist
    output_folders = [pupilSize_folder, Rco_scatter_folder, Rci_scatter_folder, Lco_scatter_folder, Lci_scatter_folder, Rco_rvalVsDelay_folder, Rci_rvalVsDelay_folder, Lco_rvalVsDelay_folder, Lci_rvalVsDelay_folder, normedMeanPupilSizes_folder, pupilSizeVsDelayLinRegress_folder]
    for output_folder in output_folders:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    return root_folder, plots_folder, monthly_mean_lums_folders, output_folders

##########################################################
def worldCam_MOIs_all_stim():
    # Moments of interest for each stimulus type (in 40ms resolution)
    all_avg_world_moments = {}
    # Stimulus 24.0
    all_avg_world_moments[24.0] = {'calibration start': {0:['2017-10','2018-05']},
    'do not move your head': {3:['2017-10','2018-05']},
    'upper left dot appears': {102:['2017-10','2017-11','2018-03']},
    'lower right dot appears': {170:['2017-10','2018-05']},
    'lower left dot appears': {238:['2017-10','2018-05']},
    'upper right dot appears': {306:['2017-10','2018-05']},
    'center dot appears': {374:['2017-10','2018-05']},
    'calibration end': {441:['2017-10','2017-11','2018-03']},
    'unique start': {442:['2017-10','2018-03','2018-05'],443:['2017-11']},
    'cat appears': {463:['2017-10','2018-01','2018-05'], 464:['2017-11']},
    'cat front paws visible': {473:['2017-10','2018-01','2018-05'], 474:['2017-11']},
    'cat lands on toy': {513:['2017-10'], 514:['2018-05']},
    'cat back paws bounce': {549:['2017-10'],547:['2018-05']},
    'unique end': {596:['2017-10','2017-11'],598:['2018-03']},
    'octo start': {595:['2017-10','2018-03'],596:['2017-11']},
    'fish turns': {645:['2017-10','2018-05']},
    'octopus fully decamouflaged': {766:['2018-05'], 767:['2017-10']},
    'camera zooms in on octopus': {860:['2017-10','2018-05']},
    'octopus inks': {882:['2017-10'],883:['2017-11','2018-03']},
    'camera clears ink cloud': {916:['2017-10'],920:['2018-05']},
    'octo end': {987:['2017-10'],989:['2017-11'],990:['2018-03']}}
    # Stimulus 25.0
    all_avg_world_moments[25.0] = {'calibration start': {0:['2017-10','2017-11','2018-03']},
    'do not move your head': {3:['2017-10','2018-05']},
    'upper left dot appears': {102:['2017-10','2017-11','2018-03']},
    'lower right dot appears': {170:['2017-10','2018-05']},
    'lower left dot appears': {239:['2017-10'],238:['2018-05']},
    'upper right dot appears': {307:['2017-10'],306:['2018-05']},
    'center dot appears': {375:['2017-10'],374:['2018-05']},
    'calibration end': {441:['2017-10','2017-11','2018-03']},
    'unique start': {442:['2018-03'],443:['2017-10','2017-11']},
    'fingers appear': {443:['2017-10'], 442:['2018-05']},
    'bird flies towards fingers': {462:['2018-05'],463:['2017-10']},
    'beak contacts food': {491:['2017-10'],492:['2018-05']},
    'wings at top of frame': {535:['2017-10','2018-05']},
    'bird flutters': {553:['2017-10'], 553:['2018-05']},
    'bird lands': {561:['2017-10'], 562:['2018-05']},
    'bird flies past fingers': {573:['2017-10','2018-05']},
    'unique end': {599:['2017-10'],600:['2017-11'],601:['2018-03']},
    'octo start': {599:['2017-10','2017-11','2018-03']},
    'fish turns': {649:['2017-10','2018-05']},
    'octopus fully decamouflaged': {770:['2017-10','2018-05']},
    'camera zooms in on octopus': {863:['2018-05'],864:['2017-10']},
    'octopus inks': {885:['2017-10','2018-03'],886:['2017-11']},
    'camera clears ink cloud': {919:['2017-10'],923:['2018-05']},
    'octo end': {989:['2017-10'],993:['2017-11'],994:['2018-03']}}
    # Stimulus 26.0
    all_avg_world_moments[26.0] = {'calibration start': {0:['2017-10','2017-11','2018-03']},
    'do not move your head': {2:['2018-05'],3:['2017-10']},
    'upper left dot appears': {102:['2017-10','2017-11','2018-03']},
    'lower right dot appears': {170:['2017-10','2018-05']},
    'lower left dot appears': {238:['2017-10','2018-05']},
    'upper right dot appears': {306:['2017-10','2018-05']},
    'center dot appears': {374:['2017-10','2018-05']},
    'calibration end': {441:['2017-10','2017-11','2018-03']},
    'unique start': {442:['2017-10','2018-03'],443:['2017-11']},
    'eyespots appear': {449:['2017-10', '2018-05']},
    'eyespots disappear, eyes darken': {487:['2017-10','2018-05']},
    'arms spread': {533:['2017-10'], 534:['2018-05']},
    'arms in, speckled mantle': {558:['2017-10'], 561:['2018-05']},
    'unique end': {663:['2017-10'],665:['2017-11','2018-03']},
    'octo start': {662:['2017-10'],663:['2018-03'],664:['2017-11']},
    'fish turns': {712:['2017-10','2018-05']},
    'octopus fully decamouflaged': {833:['2017-10','2018-05']},
    'camera zooms in on octopus': {927:['2017-10','2018-05']},
    'octopus inks': {949:['2017-10'],951:['2017-11','2018-03']},
    'camera clears ink cloud': {983:['2017-10'],987:['2018-05']},
    'octo end': {1054:['2017-10'],1059:['2017-11','2018-03']}}
    # Stimulus 27.0
    all_avg_world_moments[27.0] = {'calibration start': {0:['2017-10','2017-11','2018-03']},
    'do not move your head': {3:['2017-10','2018-05']},
    'upper left dot appears': {102:['2017-10','2017-11','2018-03']},
    'lower right dot appears': {170:['2017-10','2018-05']},
    'lower left dot appears': {238:['2017-10','2018-05']},
    'upper right dot appears': {306:['2018-05'],307:['2017-10']},
    'center dot appears': {374:['2018-05'],375:['2017-10']},
    'calibration end': {441:['2017-10','2017-11','2018-03']},
    'unique start': {443:['2017-10','2017-11','2018-03']},
    'cuttlefish appears': {443:['2017-10','2018-05']},
    'tentacles go ballistic': {530:['2017-10','2018-05']},
    'unique end': {606:['2017-10'],607:['2017-11','2018-03']},
    'octo start': {605:['2017-10','2017-11'],606:['2018-03']},
    'fish turns': {655:['2017-10','2018-05']},
    'octopus fully decamouflaged': {776:['2017-10','2018-05']},
    'camera zooms in on octopus': {869:['2018-05'],870:['2017-10']},
    'octopus inks': {892:['2017-10'],893:['2017-11','2018-03']},
    'camera clears ink cloud': {926:['2017-10'],929:['2018-05']},
    'octo end': {996:['2017-10'],1000:['2017-11','2018-03']}}
    # Stimulus 28.0
    all_avg_world_moments[28.0] = {'calibration start': {0:['2017-10','2017-11','2018-03']},
    'do not move your head': {2:['2018-05'],3:['2017-10']},
    'upper left dot appears': {102:['2017-10','2017-11','2018-03']},
    'lower right dot appears': {170:['2017-10','2018-05']},
    'lower left dot appears': {238:['2017-10','2018-05']},
    'upper right dot appears': {306:['2017-10','2018-05']},
    'center dot appears': {374:['2018-05'],375:['2017-10']},
    'calibration end': {441:['2017-10','2017-11','2018-03']},
    'unique start': {442:['2018-03'],443:['2017-10','2017-11']},
    'fish scatter': {456:['2017-10','2018-04','2018-10']},
    'center fish turns': {469:['2017-10'], 470:['2018-04'], 471:['2018-10']},
    'center fish swims to left': {494:['2018-04','2018-10'], 495:['2017-10']},
    'camera clears red ferns': {503:['2017-10'],506:['2018-04'],509:['2018-10']},
    'unique end': {662:['2017-10'],663:['2017-11'],666:['2018-03']},
    'octo start': {661:['2017-10'],662:['2018-03'],663:['2017-11']},
    'fish turns': {711:['2017-10','2018-05']},
    'octopus fully decamouflaged': {832:['2017-10'],834:['2018-05']},
    'camera zooms in on octopus': {927:['2017-10','2018-05']},
    'octopus inks': {948:['2017-10'],950:['2017-11','2018-03']},
    'camera clears ink cloud': {982:['2017-10'],986:['2018-05']},
    'octo end': {1054:['2017-10'],1056:['2017-11'],1059:['2018-03']}}
    # Stimulus 29.0
    all_avg_world_moments[29.0] = {'calibration start': {0:['2017-10','2017-11','2018-03']},
    'do not move your head': {3:['2017-10','2018-05']},
    'upper left dot appears': {102:['2017-10','2017-11','2018-03']},
    'lower right dot appears': {170:['2017-10','2018-05']},
    'lower left dot appears': {238:['2017-10','2018-05']},
    'upper right dot appears': {306:['2017-10','2018-05']},
    'center dot appears': {374:['2017-10','2018-05']},
    'calibration end': {441:['2017-10','2017-11','2018-03']},
    'unique start': {442:['2017-10'],443:['2017-11','2018-03']},
    'fish 1 appears': {457:['2017-10','2018-05']},
    'fish 1 turns': {495:['2017-10','2018-05']}, 
    'fish 2 appears': {538:['2017-10','2018-05']},
    'fish 2 touches mirror image': {646:['2017-10','2018-05']},
    'fish 2 disappears': {661:['2017-10','2018-05']}, 
    'fish 1 touches mirror image': {685:['2017-10','2018-05']},
    'fish 1 disappears': {702:['2017-10','2018-05']}, 
    'unique end': {717:['2017-10','2017-11'],718:['2018-03']},
    'octo start': {716:['2017-10','2018-03'],717:['2017-11']},
    'fish turns': {766:['2017-10','2018-03']},
    'octopus fully decamouflaged': {887:['2017-10','2018-05']},
    'camera zooms in on octopus': {981:['2017-10','2018-05']},
    'octopus inks': {1003:['2017-10'],1004:['2017-11','2018-03']},
    'camera clears ink cloud': {1037:['2017-10'],1041:['2018-05']},
    'octo end': {1108:['2017-10'],1110:['2017-11'],1112:['2018-03']}}
    return all_avg_world_moments

def extract_MOI_tb_downsample_corrected(all_MOI_dict, stim_num, MOI_name, start_or_end, downsample_mult):
    MOI_timebuckets = []
    for key in all_MOI_dict[stim_num][MOI_name]:
        MOI_timebuckets.append(key)
    if start_or_end == 'start':
        MOI_timebucket = np.min(MOI_timebuckets)
    if start_or_end == 'end':
        MOI_timebucket = np.max(MOI_timebuckets)
    if start_or_end != 'start' and start_or_end != 'end':
        logging.warning('Incorrect input for parameter start_or_end! Current input: %s' % (start_or_end))
    return MOI_timebucket*downsample_mult

def trim_phase_extractions(all_weighted_raw_extractions):
    min_len_extraction = np.inf
    for extraction in all_weighted_raw_extractions:
        if len(extraction) < min_len_extraction:
            min_len_extraction = len(extraction)
    trimmed_all_weighted_raw_extractions = []
    for extraction in all_weighted_raw_extractions:
        trimmed_all_weighted_raw_extractions.append(extraction[:min_len_extraction])
    return trimmed_all_weighted_raw_extractions

def calculate_weighted_mean_lum(all_weighted_raw_lums, all_weights):
    trimmed_all_weighted_raw_lums = trim_phase_extractions(all_weighted_raw_lums)
    trimmed_all_weights = trim_phase_extractions(all_weights)
    summed_lums = np.sum(trimmed_all_weighted_raw_lums, axis=0)
    summed_weights = np.sum(trimmed_all_weights, axis=0)
    weighted_mean = summed_lums / summed_weights
    return weighted_mean

def downsample_mean_raw_live_stims(mean_RL_array, downsample_mult):
    downsampled_mean_RL = []
    for i in range(0,len(mean_RL_array), downsample_mult):
        if (i+downsample_mult-1) > len(mean_RL_array):
            this_chunk_mean = np.nanmean(mean_RL_array[i:len(mean_RL_array)])
        else:
            this_chunk_mean = np.nanmean(mean_RL_array[i:i+downsample_mult-1])
        downsampled_mean_RL.append(this_chunk_mean)
    return np.array(downsampled_mean_RL)

##########################################################
# BEGIN SCRIPT
##########################################################
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", nargs='?', default="check_string_for_empty")
    parser.add_argument("--loc", nargs='?', default='laptop')
    args = parser.parse_args()
    ###################################
    # SOURCE DATA AND OUTPUT FILE LOCATIONS 
    ###################################
    root_folder, plots_folder, monthly_mean_lums_folders, output_folders = load_data(args.loc)
    pupilSize_folder = output_folders[0]
    Rco_scatter_folder = output_folders[1]
    Rci_scatter_folder = output_folders[2]
    Lco_scatter_folder = output_folders[3]
    Lci_scatter_folder = output_folders[4]
    Rco_rvalVsDelay_folder = output_folders[5]
    Rci_rvalVsDelay_folder = output_folders[6]
    Lco_rvalVsDelay_folder = output_folders[7]
    Lci_rvalVsDelay_folder = output_folders[8]
    normedMeanPupilSizes_folder = output_folders[9]
    pupilSizeVsDelayLinRegress_folder = output_folders[10]
    logging.info('ROOT FOLDER: %s \n PLOTS FOLDER: %s \n MONTHLY MEAN RAW LIVE AND WORLD CAM STIMULI DATA FOLDER: %s' % (root_folder, plots_folder, monthly_mean_lums_folders))
    print('ROOT FOLDER: %s \n PLOTS FOLDER: %s \n MONTHLY MEAN RAW LIVE AND WORLD CAM STIMULI DATA FOLDER: %s' % (root_folder, plots_folder, monthly_mean_lums_folders))
    ###################################
    # TIMING/SAMPLING VARIABLES FOR DATA EXTRACTION
    ###################################
    # downsample = collect data from every 40ms or other multiples of 20
    downsampled_bucket_size_ms = 40
    original_bucket_size_in_ms = 4
    max_length_of_stim_vid = 60000 # milliseconds
    no_of_time_buckets = max_length_of_stim_vid/original_bucket_size_in_ms
    downsampled_no_of_time_buckets = max_length_of_stim_vid/downsampled_bucket_size_ms
    downsample_multiplier = int(downsampled_bucket_size_ms/original_bucket_size_in_ms)
    ###################################
    # STIMULI VID INFO
    ###################################
    stim_vids = [24.0, 25.0, 26.0, 27.0, 28.0, 29.0]
    stim_name_to_float = {"Stim24": 24.0, "Stim25": 25.0, "Stim26": 26.0, "Stim27": 27.0, "Stim28": 28.0, "Stim29": 29.0}
    stim_float_to_name = {24.0: "Stim24", 25.0: "Stim25", 26.0: "Stim26", 27.0: "Stim27", 28.0: "Stim28", 29.0: "Stim29"}
    phase_names = ['calib', 'octo', 'unique1', 'unique2', 'unique3', 'unique4', 'unique5', 'unique6']
    ########################################################
    # COLLECT TIMING INFO FOR CALIB, OCTO, AND UNIQUE PHASES
    ########################################################
    all_avg_world_moments = worldCam_MOIs_all_stim() # these are in 40ms timebuckets
    # convert start and end timebuckets into 4ms resolution timebuckets
    do_not_move_start = {key:{} for key in stim_vids}
    do_not_move_end = {key:{} for key in stim_vids}
    pulsing_dots_start = {key:{} for key in stim_vids}
    pulsing_dots_end = {key:{} for key in stim_vids}
    uniques_start = {key:{} for key in stim_vids}
    uniques_end = {key:{} for key in stim_vids}
    octo_start = {key:{} for key in stim_vids}
    octo_end = {key:{} for key in stim_vids}
    for stim_type in stim_vids:
        # world cam
        dnm_start_world = extract_MOI_tb_downsample_corrected(all_avg_world_moments, stim_type, 'do not move your head', 'start', downsample_multiplier)
        dnm_end_world = extract_MOI_tb_downsample_corrected(all_avg_world_moments, stim_type, 'upper left dot appears', 'end', downsample_multiplier)
        pd_start_world = extract_MOI_tb_downsample_corrected(all_avg_world_moments, stim_type, 'upper left dot appears', 'start', downsample_multiplier)
        pd_end_world = extract_MOI_tb_downsample_corrected(all_avg_world_moments, stim_type, 'calibration end', 'end', downsample_multiplier)
        u_start_world = extract_MOI_tb_downsample_corrected(all_avg_world_moments, stim_type, 'unique start', 'start', downsample_multiplier)
        u_end_world = extract_MOI_tb_downsample_corrected(all_avg_world_moments, stim_type, 'unique end', 'end', downsample_multiplier)
        o_start_world = extract_MOI_tb_downsample_corrected(all_avg_world_moments, stim_type, 'octo start', 'start', downsample_multiplier)
        o_end_world = extract_MOI_tb_downsample_corrected(all_avg_world_moments, stim_type, 'octo end', 'end', downsample_multiplier)
        do_not_move_start[stim_type]['world'] = dnm_start_world
        do_not_move_end[stim_type]['world'] = dnm_end_world
        pulsing_dots_start[stim_type]['world'] = pd_start_world
        pulsing_dots_end[stim_type]['world'] = pd_end_world
        uniques_start[stim_type]['world'] = u_start_world
        uniques_end[stim_type]['world'] = u_end_world
        octo_start[stim_type]['world'] = o_start_world
        octo_end[stim_type]['world'] = o_end_world
        # raw live
        dnm_start_raw = dnm_start_world - dnm_start_world
        dnm_end_raw = dnm_end_world - dnm_start_world
        pd_start_raw = pd_start_world - dnm_start_world
        pd_end_raw = pd_end_world - dnm_start_world
        u_start_raw = u_start_world - dnm_start_world
        u_end_raw = u_end_world - dnm_start_world
        o_start_raw = o_start_world - dnm_start_world
        o_end_raw = o_end_world - dnm_start_world
        do_not_move_start[stim_type]['raw'] = dnm_start_raw
        do_not_move_end[stim_type]['raw'] = dnm_end_raw
        pulsing_dots_start[stim_type]['raw'] = pd_start_raw
        pulsing_dots_end[stim_type]['raw'] = pd_end_raw
        uniques_start[stim_type]['raw'] = u_start_raw
        uniques_end[stim_type]['raw'] = u_end_raw
        octo_start[stim_type]['raw'] = o_start_raw
        octo_end[stim_type]['raw'] = o_end_raw

    ###############################################################
    # Load mean monthly raw live stim and world cam luminance files
    # split into Do Not Move, Pulsing Dots, Unique, and Octo phases
    # calculate weighted mean of each phase
    ###############################################################
    # raw live stim
    all_weighted_raw_doNotMove = []
    all_weights_raw_doNotMove = []
    all_weighted_raw_pulsingDots = []
    all_weights_raw_pulsingDots = []
    all_weighted_raw_unique = {key:[] for key in stim_vids}
    all_weights_raw_unique = {key:[] for key in stim_vids}
    all_weighted_raw_octo = []
    all_weights_raw_octo = []
    # world cam
    all_weighted_world_keyFrames = {key:{} for key in stim_vids}
    # collect length of each stimulus type in 4ms resolution
    supersampled_length_all_stims = {key:[] for key in stim_vids}
    # extract raw live stim and world cam data
    for monthly_mean_folder in monthly_mean_lums_folders:
        raw_live_stim_files = glob.glob(root_folder + os.sep + monthly_mean_folder + os.sep + '*_meanRawLiveStim_*.npy')
        world_cam_files = glob.glob(root_folder + os.sep + monthly_mean_folder + os.sep + '*_meanWorldCam_*.npy')
        # raw live - extract and split into phases
        for raw_live_stim in raw_live_stim_files:
            stim_type = stim_name_to_float[os.path.basename(raw_live_stim).split('_')[1]]
            vid_count = float(os.path.basename(raw_live_stim).split('_')[-1][:-8])
            raw_live_array = np.load(raw_live_stim)
            supersampled_length_all_stims[stim_type].append(len(raw_live_array))
            this_file_weighted_doNotMove = []
            this_file_weights_doNotMove = []
            this_file_weighted_pulsingDots = []
            this_file_weights_pulsingDots = []
            this_file_weighted_unique = []
            this_file_weights_unique = []
            this_file_weighted_octo = []
            this_file_weights_octo = []
            for row in raw_live_array:
                timebucket = row[0]
                weight = row[1]
                mean_lum = row[2]
                this_tb_weighted_lum = weight*mean_lum
                if do_not_move_start[stim_type]['raw'] < timebucket < do_not_move_end[stim_type]['raw']:
                    this_file_weighted_doNotMove.append(this_tb_weighted_lum)
                    this_file_weights_doNotMove.append(weight)
                    continue
                elif pulsing_dots_start[stim_type]['raw'] < timebucket < pulsing_dots_end[stim_type]['raw']:
                    this_file_weighted_pulsingDots.append(this_tb_weighted_lum)
                    this_file_weights_pulsingDots.append(weight)
                elif uniques_start[stim_type]['raw'] < timebucket < uniques_end[stim_type]['raw']:
                    this_file_weighted_unique.append(this_tb_weighted_lum)
                    this_file_weights_unique.append(weight)
                elif octo_start[stim_type]['raw'] < timebucket < octo_end[stim_type]['raw']:
                    this_file_weighted_octo.append(this_tb_weighted_lum)
                    this_file_weights_octo.append(weight)
            all_weighted_raw_doNotMove.append(np.array(this_file_weighted_doNotMove))
            all_weights_raw_doNotMove.append(np.array(this_file_weights_doNotMove))
            all_weighted_raw_pulsingDots.append(np.array(this_file_weighted_pulsingDots))
            all_weights_raw_pulsingDots.append(np.array(this_file_weights_pulsingDots))
            all_weighted_raw_unique[stim_type].append(np.array(this_file_weighted_unique))
            all_weights_raw_unique[stim_type].append(np.array(this_file_weights_unique))
            all_weighted_raw_octo.append(np.array(this_file_weighted_octo))
            all_weights_raw_octo.append(np.array(this_file_weights_octo))
        # world cam - extract for SANITY CHECK
        for world_cam in world_cam_files:
            stim_type = stim_name_to_float[os.path.basename(world_cam).split('_')[1]]
            vid_count = float(os.path.basename(world_cam).split('_')[-1][:-8])
            world_cam_frames = np.load(world_cam)
            for key_frame in world_cam_frames:
                timebucket = key_frame[0]
                weight = key_frame[1]
                mean_frame = key_frame[2]
                this_tb_weighted_frame = weight*mean_frame
                if timebucket in all_weighted_world_keyFrames[stim_type].keys():
                    all_weighted_world_keyFrames[stim_type][timebucket]['weighted frames'].append(this_tb_weighted_frame)
                    all_weighted_world_keyFrames[stim_type][timebucket]['weights'].append(weight)
                else:
                    all_weighted_world_keyFrames[stim_type][timebucket] = {'weighted frames':[this_tb_weighted_frame], 'weights':[weight]}

    # mean raw live luminance arrays
    mean_raw_live_doNotMove = calculate_weighted_mean_lum(all_weighted_raw_doNotMove, all_weights_raw_doNotMove)
    mean_raw_live_pulsingDots = calculate_weighted_mean_lum(all_weighted_raw_pulsingDots, all_weights_raw_pulsingDots)
    mean_raw_live_calib = np.concatenate([mean_raw_live_doNotMove, mean_raw_live_pulsingDots])
    mean_raw_live_uniques = {key:None for key in stim_vids}
    for stim in all_weighted_raw_unique:
        mean_raw_live_uniques[stim] = calculate_weighted_mean_lum(all_weighted_raw_unique[stim], all_weights_raw_unique[stim])
    mean_raw_live_octo = calculate_weighted_mean_lum(all_weighted_raw_octo, all_weights_raw_octo)

    #####################################################################################################
    # WORLD CAM SANITY CHECK: calculate mean frame luminance from world cams and compare to raw live stim
    # Save monthly mean world cam videos for each stim type 
    # THIS SECTION UNDER CONSTRUCTION!!
    ######################################################################################################
    # fill in gaps between keyframes
    allStim_weighted_mean_world_allFrames = {key:{} for key in stim_vids}
    for stim in all_weighted_world_keyFrames.keys():
        ordered_keyframes = sorted(all_weighted_world_keyFrames[stim].keys())
        this_stim_all_weighted_frames = []
        this_stim_all_weights = []
        for i, keyframe in enumerate(ordered_keyframes):
            if i==0 and keyframe==0:
                this_stim_all_weighted_frames.append(all_weighted_world_keyFrames[stim][keyframe]['weighted frames'])
                this_stim_all_weights.append(all_weighted_world_keyFrames[stim][keyframe]['weights'])
            elif i==0 and keyframe!=0:
                for frame in range(keyframe-1):
                    this_stim_all_weighted_frames.append(np.nan)
                    this_stim_all_weights.append(np.nan)
                this_stim_all_weighted_frames.append(all_weighted_world_keyFrames[stim][keyframe]['weighted frames'])
                this_stim_all_weights.append(all_weighted_world_keyFrames[stim][keyframe]['weights'])
            else:
                prev_keyframe = ordered_keyframes[i-1]
                if keyframe - prev_keyframe > 1:
                    for frame in range(prev_keyframe, keyframe-1):
                        this_stim_all_weighted_frames.append(all_weighted_world_keyFrames[stim][prev_keyframe]['weighted frames'])
                        this_stim_all_weights.append(all_weighted_world_keyFrames[stim][prev_keyframe]['weights'])
                this_stim_all_weighted_frames.append(all_weighted_world_keyFrames[stim][keyframe]['weighted frames'])
                this_stim_all_weights.append(all_weighted_world_keyFrames[stim][keyframe]['weights'])
        full_length_this_stim = np.min(supersampled_length_all_stims[stim])
        if ordered_keyframes[-1] < full_length_this_stim:
            for frame in range(ordered_keyframes[-1], full_length_this_stim):
                this_stim_all_weighted_frames.append(all_weighted_world_keyFrames[stim][ordered_keyframes[-1]]['weighted frames'])
                this_stim_all_weights.append(all_weighted_world_keyFrames[stim][ordered_keyframes[-1]]['weights'])
        this_stim_weighted_timebuckets = []
        this_stim_weights = []
        for timebucket in this_stim_all_weighted_frames:
            this_tb_weighted_frame = np.sum(np.array(timebucket), axis=0)
            this_stim_weighted_timebuckets.append(this_tb_weighted_frame)
        for timebucket in this_stim_all_weights:
            this_tb_weights = np.sum(np.array(timebucket), axis=0)
            this_stim_weights.append(this_tb_weights)
        allStim_weighted_mean_world_allFrames[stim]['weighted timebuckets'] = this_stim_weighted_timebuckets
        allStim_weighted_mean_world_allFrames[stim]['weights'] = this_stim_weights
    # split into phases
    all_weighted_world_frames_calib = {'weighted timebuckets':[], 'weights':[]}
    all_weighted_world_frames_uniques = {key:{'weighted timebuckets':[], 'weights':[]} for key in stim_vids}
    all_weighted_world_frames_octo = {'weighted timebuckets':[], 'weights':[]}
    for stim in allStim_weighted_mean_world_allFrames.keys():
        this_stim_weighted_doNotMove = []
        this_stim_weights_doNotMove = []
        this_stim_weighted_pulsingDots = []
        this_stim_weights_pulsingDots = []
        this_stim_weighted_unique = []
        this_stim_weights_unique = []
        this_stim_weighted_octo = []
        this_stim_weights_octo = []
        for tb, weighted_frame in enumerate(allStim_weighted_mean_world_allFrames[stim]['weighted timebuckets']):
            this_tb_weight = allStim_weighted_mean_world_allFrames[stim]['weights'][tb]
            if do_not_move_start[stim_type]['world'] < tb < do_not_move_end[stim_type]['world']:
                this_stim_weighted_doNotMove.append(weighted_frame)
                this_stim_weights_doNotMove.append(this_tb_weight)
            elif pulsing_dots_start[stim_type]['world'] < tb < pulsing_dots_end[stim_type]['world']:
                this_stim_weighted_pulsingDots.append(weighted_frame)
                this_stim_weights_pulsingDots.append(this_tb_weight)
            elif uniques_start[stim_type]['world'] < timebucket < uniques_end[stim_type]['world']:
                this_stim_weighted_unique.append(weighted_frame)
                this_stim_weights_unique.append(this_tb_weight)
            elif octo_start[stim_type]['world'] < tb < octo_end[stim_type]['world']:
                this_stim_weighted_octo.append(weighted_frame)
                this_stim_weights_octo.append(this_tb_weight)
        this_stim_weighted_calib = np.concatenate((this_stim_weighted_doNotMove, this_stim_weighted_pulsingDots))
        this_stim_weights_calib = np.concatenate((this_stim_weights_doNotMove, this_stim_weights_pulsingDots))
        all_weighted_world_frames_calib['weighted timebuckets'].append(this_stim_weighted_calib)
        all_weighted_world_frames_calib['weights'].append(this_stim_weights_calib)
        all_weighted_world_frames_uniques[stim]['weighted timebuckets'].append(this_stim_weighted_unique)
        all_weighted_world_frames_uniques[stim]['weights'].append(this_stim_weights_unique)
        all_weighted_world_frames_octo['weighted timebuckets'].append(this_stim_weighted_octo)
        all_weighted_world_frames_octo['weights'].append(this_stim_weights_octo)



all_weighted_raw_calib = {'weighted timebuckets':[], 'weights':[]}
all_weighted_raw_unique = {key:{'weighted timebuckets':[], 'weights':[]} for key in stim_vids}
all_weighted_raw_octo = {'weighted timebuckets':[], 'weights':[]}

# raw live - extract and split into phases
for raw_live_stim in raw_live_stim_files:
    stim_type = stim_name_to_float[os.path.basename(raw_live_stim).split('_')[1]]
    vid_count = float(os.path.basename(raw_live_stim).split('_')[-1][:-8])
    raw_live_array = np.load(raw_live_stim)
    supersampled_length_all_stims[stim_type].append(len(raw_live_array))


def split_into_stim_phases(full_stim_array):

    this_weighted_doNotMove = []
    this_weights_doNotMove = []
    this_weighted_pulsingDots = []
    this_weights_pulsingDots = []
    this_weighted_unique = []
    this_weights_unique = []
    this_weighted_octo = []
    this_weights_octo = []
    for row in raw_live_array:
        timebucket = row[0]
        weight = row[1]
        mean_lum = row[2]
        this_tb_weighted_lum = weight*mean_lum
        if do_not_move_start[stim_type]['raw'] < timebucket < do_not_move_end[stim_type]['raw']:
            this_file_weighted_doNotMove.append(this_tb_weighted_lum)
            this_file_weights_doNotMove.append(weight)
            continue
        elif pulsing_dots_start[stim_type]['raw'] < timebucket < pulsing_dots_end[stim_type]['raw']:
            this_file_weighted_pulsingDots.append(this_tb_weighted_lum)
            this_file_weights_pulsingDots.append(weight)
        elif uniques_start[stim_type]['raw'] < timebucket < uniques_end[stim_type]['raw']:
            this_file_weighted_unique.append(this_tb_weighted_lum)
            this_file_weights_unique.append(weight)
        elif octo_start[stim_type]['raw'] < timebucket < octo_end[stim_type]['raw']:
            this_file_weighted_octo.append(this_tb_weighted_lum)
            this_file_weights_octo.append(weight)
    all_weighted_raw_doNotMove.append(np.array(this_file_weighted_doNotMove))
    all_weights_raw_doNotMove.append(np.array(this_file_weights_doNotMove))
    all_weighted_raw_pulsingDots.append(np.array(this_file_weighted_pulsingDots))
    all_weights_raw_pulsingDots.append(np.array(this_file_weights_pulsingDots))
    all_weighted_raw_unique[stim_type].append(np.array(this_file_weighted_unique))
    all_weights_raw_unique[stim_type].append(np.array(this_file_weights_unique))
    all_weighted_raw_octo.append(np.array(this_file_weighted_octo))
    all_weights_raw_octo.append(np.array(this_file_weights_octo))







        
    # calculate weighted mean frame 
    all_stims_summed_weighted_frames_world_calib = np.sum(np.array(all_weighted_world_frames_calib['weighted timebuckets']), axis=0)
    all_stims_summed_weights_world_calib = np.sum(np.array(all_weighted_world_frames_calib['weights']), axis=0)
    weighted_mean_world_frames_calib = []
    for tb, weight in enumerate(all_stims_summed_weights_world_calib):
        this_tb_weighted_mean_frame = all_stims_summed_weighted_frames_world_calib[tb]/weight
        weighted_mean_world_frames_calib.append(this_tb_weighted_mean_frame)
    
    
    # downsample framerate to 30fps
    downsampled_mean_world = []
    for i in range(0,len(weighted_mean_world_frames_calib), downsample_multiplier):
        if (i+downsample_multiplier-1) > len(weighted_mean_world_frames_calib):
            this_chunk_mean = np.nanmean(weighted_mean_world_frames_calib[i:len(weighted_mean_world_frames_calib)], axis=0)
        else:
            this_chunk_mean = np.nanmean(weighted_mean_world_frames_calib[i:i+downsample_multiplier-1], axis=0)
        downsampled_mean_world.append(this_chunk_mean)






    # reshape into original world cam dimensions
    weighted_mean_world_frames_calib_reshaped = []
    for frame in downsampled_mean_world:
        reshaped_frame = np.reshape(frame,(120,160))
        weighted_mean_world_frames_calib_reshaped.append(reshaped_frame)
    
    # make mean world cam movie for this phase and save as .mp4 file
    write_path = 'test.mp4'
    end_tbucket = len(weighted_mean_world_frames_calib_reshaped)
    # temporarily switch matplotlib backend in order to write video
    plt.switch_backend("Agg")
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    FF_writer = animation.FFMpegWriter(fps=25, codec='h264', metadata=dict(artist='Danbee Kim'))
    fig = plt.figure()
    i = 0
    im = plt.imshow(weighted_mean_world_frames_calib_reshaped[i], cmap='gray', animated=True)
    def updatefig(*args):
        global i
        if (i<end_tbucket):
            i += 1
        else:
            i=0
        im.set_array(weighted_mean_world_frames_calib_reshaped[i])
        return im,
    ani = animation.FuncAnimation(fig, updatefig, frames=len(weighted_mean_world_frames_calib_reshaped), interval=50, blit=True)
    print("Writing average world video frames to {path}...".format(path=write_path))
    ani.save(write_path, writer=FF_writer)
    plt.close(fig)
    print("Finished writing!")
    # restore default matplotlib backend
    plt.switch_backend('TkAgg')




    weighted_mean_world_frames_uniques
    weighted_mean_world_frames_octo
    
    # convert into a single lum value per frame



def write_avg_world_vid(avg_world_vid_tbucketed_dict, start_tbucket, end_tbucket, write_path):
    # temporarily switch matplotlib backend in order to write video
    plt.switch_backend("Agg")
    # convert dictionary of avg world vid frames into a list of arrays
    tbucket_frames = []
    sorted_tbuckets = sorted([x for x in avg_world_vid_tbucketed_dict.keys() if type(x) is int])
    for tbucket in sorted_tbuckets:
        tbucket_frames.append(avg_world_vid_tbucketed_dict[tbucket])
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    FF_writer = animation.FFMpegWriter(fps=30, codec='h264', metadata=dict(artist='Danbee Kim'))
    fig = plt.figure()
    i = start_tbucket
    im = plt.imshow(tbucket_frames[i], cmap='gray', animated=True)
    def updatefig(*args):
        global i
        if (i<end_tbucket):
            i += 1
        else:
            i=0
        im.set_array(tbucket_frames[i])
        return im,
    ani = animation.FuncAnimation(fig, updatefig, frames=len(tbucket_frames), interval=50, blit=True)
    print("Writing average world video frames to {path}...".format(path=write_path))
    ani.save(write_path, writer=FF_writer)
    plt.close(fig)
    print("Finished writing!")
    # restore default matplotlib backend
    plt.switch_backend('TkAgg')


############################################################################
# Downsample raw live stim data to match pupil data (4ms to 40ms resolution)
# BEFORE DOWNSAMPLING, NEED TO ADJUST FOR WORLD CAM CAPTURING 2-3 FRAMES OF "PLEASE CENTER EYE" PHASE
# MEASURING THE DISPLAY LATENCY WILL BE A SEPARATE SCRIPT
############################################################################
downsampled_mean_RL_calib = downsample_mean_raw_live_stims(mean_raw_live_calib, downsample_multiplier)
downsampled_mean_RL_octo = downsample_mean_raw_live_stims(mean_raw_live_octo, downsample_multiplier)
downsampled_mean_RL_uniques = {key:None for key in stim_vids}
for stim in mean_raw_live_uniques:
    downsampled_mean_RL_uniques[stim] = downsample_mean_raw_live_stims(mean_raw_live_uniques[stim], downsample_multiplier)

downsampled_mean_RL_all_phases = [downsampled_mean_RL_calib, downsampled_mean_RL_octo, downsampled_mean_RL_uniques[24.0], downsampled_mean_RL_uniques[25.0], downsampled_mean_RL_uniques[26.0], downsampled_mean_RL_uniques[27.0], downsampled_mean_RL_uniques[28.0], downsampled_mean_RL_uniques[29.0]]
calib_len = len(downsampled_mean_RL_calib)
octo_len = len(downsampled_mean_RL_octo)
unique_lens = [len(downsampled_mean_RL_uniques[24.0]), len(downsampled_mean_RL_uniques[25.0]), len(downsampled_mean_RL_uniques[26.0]), len(downsampled_mean_RL_uniques[27.0]), len(downsampled_mean_RL_uniques[28.0]), len(downsampled_mean_RL_uniques[29.0])]

###################################
# BEGIN PUPIL DATA EXTRACTION 
###################################
# prepare to sort pupil data by stimulus
all_right_trials_contours_X = {key:[] for key in stim_vids}
all_right_trials_contours_Y = {key:[] for key in stim_vids}
all_right_trials_contours = {key:[] for key in stim_vids}
all_right_trials_circles_X = {key:[] for key in stim_vids}
all_right_trials_circles_Y = {key:[] for key in stim_vids}
all_right_trials_circles = {key:[] for key in stim_vids}
all_left_trials_contours_X = {key:[] for key in stim_vids}
all_left_trials_contours_Y = {key:[] for key in stim_vids}
all_left_trials_contours = {key:[] for key in stim_vids}
all_left_trials_circles_X = {key:[] for key in stim_vids}
all_left_trials_circles_Y = {key:[] for key in stim_vids}
all_left_trials_circles = {key:[] for key in stim_vids}
all_trials_position_X_data = [all_right_trials_contours_X, all_right_trials_circles_X, all_left_trials_contours_X, all_left_trials_circles_X]
all_trials_position_Y_data = [all_right_trials_contours_Y, all_right_trials_circles_Y, all_left_trials_contours_Y, all_left_trials_circles_Y]
all_trials_size_data = [all_right_trials_contours, all_right_trials_circles, all_left_trials_contours, all_left_trials_circles]
activation_count = {}
analysed_count = {}
stimuli_tbucketed = {key:[] for key in stim_vids}
# consolidate csv files from multiple days into one data structure
day_folders = sorted(os.listdir(root_folder))
# find pupil data on dropbox
pupil_folders = fnmatch.filter(day_folders, 'SurprisingMinds_*')
# first day was a debugging session, so skip it
pupil_folders = pupil_folders[1:]
########################################################
if args.a == 'check_string_for_empty':
    logging.info('Extracting all pupil data...')
    print('Extracting all pupil data...')
elif args.a == 'debug':
    logging.warning('Extracting debugging subset of pupil data...')
    print('Extracting debugging subset of pupil data...')
    pupil_folders = pupil_folders[5:10]
    # if currently still running pupil finding analysis...
    pupil_folders = pupil_folders[:-1]
else:
    logging.warning('%s is not a valid optional input to this script! \nExtracting all pupil data...' % (args.a))
    print('%s is not a valid optional input to this script! \nExtracting all pupil data...' % (args.a))
########################################################
# collect dates for which pupil extraction fails
failed_days = []
for day_folder in pupil_folders:
    # for each day...
    day_folder_path = os.path.join(root_folder, day_folder)
    analysis_folder = os.path.join(day_folder_path, "Analysis")
    csv_folder = os.path.join(analysis_folder, "csv")
    world_folder = os.path.join(analysis_folder, "world")
    #
    # Print/save number of users per day
    day_name = day_folder.split("_")[-1]
    try:
        ## EXTRACT PUPIL SIZE AND POSITION
        right_area_contours_X, right_area_contours_Y, right_area_contours, right_area_circles_X, right_area_circles_Y, right_area_circles, num_right_activations, num_good_right_trials = load_daily_pupils("right", csv_folder, downsampled_no_of_time_buckets, original_bucket_size_in_ms, downsampled_bucket_size_ms)
        left_area_contours_X, left_area_contours_Y, left_area_contours, left_area_circles_X, left_area_circles_Y, left_area_circles, num_left_activations, num_good_left_trials = load_daily_pupils("left", csv_folder, downsampled_no_of_time_buckets, original_bucket_size_in_ms, downsampled_bucket_size_ms)
        #
        analysed_count[day_name] = [num_good_right_trials, num_good_left_trials]
        activation_count[day_name] = [num_right_activations, num_left_activations]
        print("On {day}, exhibit was activated {right_count} times (right) and {left_count} times (left), with {right_good_count} good right trials and {left_good_count} good left trials".format(day=day_name, right_count=num_right_activations, left_count=num_left_activations, right_good_count=num_good_right_trials, left_good_count=num_good_left_trials))
        # separate by stimulus number
        R_contours_X = {key:[] for key in stim_vids}
        R_contours_Y = {key:[] for key in stim_vids}
        R_contours = {key:[] for key in stim_vids}
        R_circles_X = {key:[] for key in stim_vids}
        R_circles_Y = {key:[] for key in stim_vids}
        R_circles = {key:[] for key in stim_vids}
        L_contours_X = {key:[] for key in stim_vids}
        L_contours_Y = {key:[] for key in stim_vids}
        L_contours = {key:[] for key in stim_vids}
        L_circles_X = {key:[] for key in stim_vids}
        L_circles_Y = {key:[] for key in stim_vids}
        L_circles = {key:[] for key in stim_vids}
        #
        stim_sorted_data_right = [R_contours_X, R_contours_Y, R_contours, R_circles_X, R_circles_Y, R_circles]
        stim_sorted_data_left = [L_contours_X, L_contours_Y, L_contours, L_circles_X, L_circles_Y, L_circles]
        stim_sorted_data_all = [stim_sorted_data_right, stim_sorted_data_left]
        #
        extracted_data_right = [right_area_contours_X, right_area_contours_Y, right_area_contours, right_area_circles_X, right_area_circles_Y, right_area_circles]
        extracted_data_left = [left_area_contours_X, left_area_contours_Y, left_area_contours, left_area_circles_X, left_area_circles_Y, left_area_circles]
        extracted_data_all = [extracted_data_right, extracted_data_left]
        #
        for side in range(len(extracted_data_all)):
            for dataset in range(len(extracted_data_all[side])):
                for trial in extracted_data_all[side][dataset]:
                    stim_num = trial[-1]
                    if stim_num in stim_sorted_data_all[side][dataset].keys():
                        stim_sorted_data_all[side][dataset][stim_num].append(trial[:-1])
        #
        # filter data for outlier points
        all_position_X_data = [R_contours_X, R_circles_X, L_contours_X, L_circles_X]
        all_position_Y_data = [R_contours_Y, R_circles_Y, L_contours_Y, L_circles_Y]
        all_size_data = [R_contours, R_circles, L_contours, L_circles]
        # remove:
        # eye positions that are not realistic
        # time buckets with no corresponding frames
        # video pixel limits are (798,599)
        all_position_X_data = filter_to_nan(all_position_X_data, 798, 0)
        all_position_Y_data = filter_to_nan(all_position_Y_data, 599, 0)
        # contours/circles that are too big
        all_size_data = filter_to_nan(all_size_data, 15000, 0)
        #
        # append position data to global data structure
        for i in range(len(all_position_X_data)):
            for stimulus in all_position_X_data[i]:
                for index in range(len(all_position_X_data[i][stimulus])):
                    all_trials_position_X_data[i][stimulus].append(all_position_X_data[i][stimulus][index])
        for i in range(len(all_position_Y_data)):
            for stimulus in all_position_Y_data[i]:
                for index in range(len(all_position_Y_data[i][stimulus])):
                    all_trials_position_Y_data[i][stimulus].append(all_position_Y_data[i][stimulus][index])
        # append size data to global data structure
        for i in range(len(all_size_data)):
            for stimulus in all_size_data[i]:
                for index in range(len(all_size_data[i][stimulus])):
                    all_trials_size_data[i][stimulus].append(all_size_data[i][stimulus][index])
        print("Day {day} succeeded!".format(day=day_name))
    except Exception:
        failed_days.append(day_name)
        print("Day {day} failed!".format(day=day_name))

###################################
# Normalize pupil size data 
###################################
# Right Contours
R_contours_allStim = [all_trials_size_data[0][24.0], all_trials_size_data[0][25.0], all_trials_size_data[0][26.0], all_trials_size_data[0][27.0], all_trials_size_data[0][28.0], all_trials_size_data[0][29.0]]
Rco_normed = normPupilSizeData(R_contours_allStim, 'right contours')
# Right Circles
R_circles_allStim = [all_trials_size_data[1][24.0], all_trials_size_data[1][25.0], all_trials_size_data[1][26.0], all_trials_size_data[1][27.0], all_trials_size_data[1][28.0], all_trials_size_data[1][29.0]]
Rci_normed = normPupilSizeData(R_circles_allStim, 'right circles')
# Left Contours
L_contours_allStim = [all_trials_size_data[2][24.0], all_trials_size_data[2][25.0], all_trials_size_data[2][26.0], all_trials_size_data[2][27.0], all_trials_size_data[2][28.0], all_trials_size_data[2][29.0]]
Lco_normed = normPupilSizeData(L_contours_allStim, 'left contours')
# Left Circles
L_circles_allStim = [all_trials_size_data[3][24.0], all_trials_size_data[3][25.0], all_trials_size_data[3][26.0], all_trials_size_data[3][27.0], all_trials_size_data[3][28.0], all_trials_size_data[3][29.0]]
Lci_normed = normPupilSizeData(L_circles_allStim, 'left circles')

###################################
# split normed pupil size arrays based on different delays of pupil reaction
# save split normed pupil size arrays as binary files
# create scatter plot of pupil size against world cam luminance values
# include least squares regression line in scatter plot
###################################
delays = 25
# by phase
Rco_calibLinRegress_allDelays = []
Rci_calibLinRegress_allDelays = []
Lco_calibLinRegress_allDelays = []
Lci_calibLinRegress_allDelays = []
Rco_octoLinRegress_allDelays = []
Rci_octoLinRegress_allDelays = []
Lco_octoLinRegress_allDelays = []
Lci_octoLinRegress_allDelays = []
Rco_u1LinRegress_allDelays = []
Rci_u1LinRegress_allDelays = []
Lco_u1LinRegress_allDelays = []
Lci_u1LinRegress_allDelays = []
Rco_u2LinRegress_allDelays = []
Rci_u2LinRegress_allDelays = []
Lco_u2LinRegress_allDelays = []
Lci_u2LinRegress_allDelays = []
Rco_u3LinRegress_allDelays = []
Rci_u3LinRegress_allDelays = []
Lco_u3LinRegress_allDelays = []
Lci_u3LinRegress_allDelays = []
Rco_u4LinRegress_allDelays = []
Rci_u4LinRegress_allDelays = []
Lco_u4LinRegress_allDelays = []
Lci_u4LinRegress_allDelays = []
Rco_u5LinRegress_allDelays = []
Rci_u5LinRegress_allDelays = []
Lco_u5LinRegress_allDelays = []
Lci_u5LinRegress_allDelays = []
Rco_u6LinRegress_allDelays = []
Rci_u6LinRegress_allDelays = []
Lco_u6LinRegress_allDelays = []
Lci_u6LinRegress_allDelays = []
# all phases
Rco_allPhasesConcatLinRegress_allDelays = []
Rci_allPhasesConcatLinRegress_allDelays = []
Lco_allPhasesConcatLinRegress_allDelays = []
Lci_allPhasesConcatLinRegress_allDelays = []
for delay in range(delays):
    print('Delay: %d timebucket(s)'%(delay))
    # Right Contours
    linRegress_Rco = splitPupils_withDelay_plotScatterLinRegress(delay, downsampled_bucket_size_ms, downsampled_mean_RL_all_phases, Rco_normed, calib_len, unique_lens, octo_len, 'RightContours', Rco_scatter_folder, normedMeanPupilSizes_folder)
    Rco_allPhasesConcatLinRegress_allDelays.append(linRegress_Rco[0])
    Rco_calibLinRegress_allDelays.append(linRegress_Rco[1])
    Rco_octoLinRegress_allDelays.append(linRegress_Rco[2])
    Rco_u1LinRegress_allDelays.append(linRegress_Rco[3])
    Rco_u2LinRegress_allDelays.append(linRegress_Rco[4])
    Rco_u3LinRegress_allDelays.append(linRegress_Rco[5])
    Rco_u4LinRegress_allDelays.append(linRegress_Rco[6])
    Rco_u5LinRegress_allDelays.append(linRegress_Rco[7])
    Rco_u6LinRegress_allDelays.append(linRegress_Rco[8])
    # Right Circles
    linRegress_Rci = splitPupils_withDelay_plotScatterLinRegress(delay, downsampled_bucket_size_ms, downsampled_mean_RL_all_phases, Rci_normed, calib_len, unique_lens, octo_len, 'RightCircles', Rci_scatter_folder, normedMeanPupilSizes_folder)
    Rci_allPhasesConcatLinRegress_allDelays.append(linRegress_Rci[0])
    Rci_calibLinRegress_allDelays.append(linRegress_Rci[1])
    Rci_octoLinRegress_allDelays.append(linRegress_Rci[2])
    Rci_u1LinRegress_allDelays.append(linRegress_Rci[3])
    Rci_u2LinRegress_allDelays.append(linRegress_Rci[4])
    Rci_u3LinRegress_allDelays.append(linRegress_Rci[5])
    Rci_u4LinRegress_allDelays.append(linRegress_Rci[6])
    Rci_u5LinRegress_allDelays.append(linRegress_Rci[7])
    Rci_u6LinRegress_allDelays.append(linRegress_Rci[8])
    # Left Contours
    linRegress_Lco = splitPupils_withDelay_plotScatterLinRegress(delay, downsampled_bucket_size_ms, downsampled_mean_RL_all_phases, Lco_normed, calib_len, unique_lens, octo_len, 'LeftContours', Lco_scatter_folder, normedMeanPupilSizes_folder)
    Lco_allPhasesConcatLinRegress_allDelays.append(linRegress_Lco[0])
    Lco_calibLinRegress_allDelays.append(linRegress_Lco[1])
    Lco_octoLinRegress_allDelays.append(linRegress_Lco[2])
    Lco_u1LinRegress_allDelays.append(linRegress_Lco[3])
    Lco_u2LinRegress_allDelays.append(linRegress_Lco[4])
    Lco_u3LinRegress_allDelays.append(linRegress_Lco[5])
    Lco_u4LinRegress_allDelays.append(linRegress_Lco[6])
    Lco_u5LinRegress_allDelays.append(linRegress_Lco[7])
    Lco_u6LinRegress_allDelays.append(linRegress_Lco[8])
    # Left Circles
    linRegress_Lci = splitPupils_withDelay_plotScatterLinRegress(delay, downsampled_bucket_size_ms, downsampled_mean_RL_all_phases, Lci_normed, calib_len, unique_lens, octo_len, 'LeftCircles', Lci_scatter_folder, normedMeanPupilSizes_folder)
    Lci_allPhasesConcatLinRegress_allDelays.append(linRegress_Lci[0])
    Lci_calibLinRegress_allDelays.append(linRegress_Lci[1])
    Lci_octoLinRegress_allDelays.append(linRegress_Lci[2])
    Lci_u1LinRegress_allDelays.append(linRegress_Lci[3])
    Lci_u2LinRegress_allDelays.append(linRegress_Lci[4])
    Lci_u3LinRegress_allDelays.append(linRegress_Lci[5])
    Lci_u4LinRegress_allDelays.append(linRegress_Lci[6])
    Lci_u5LinRegress_allDelays.append(linRegress_Lci[7])
    Lci_u6LinRegress_allDelays.append(linRegress_Lci[8])

###################################
# plot fit scores (rvals) vs delay
###################################
# all phases combined
drawFitScoresVsDelay_full(Rco_allPhasesConcatLinRegress_allDelays, delays, 'RightContours', downsampled_bucket_size_ms, Rco_rvalVsDelay_folder) 
drawFitScoresVsDelay_full(Rci_allPhasesConcatLinRegress_allDelays, delays, 'RightCircles', downsampled_bucket_size_ms, Rci_rvalVsDelay_folder) 
drawFitScoresVsDelay_full(Lco_allPhasesConcatLinRegress_allDelays, delays, 'LeftContours', downsampled_bucket_size_ms, Lco_rvalVsDelay_folder) 
drawFitScoresVsDelay_full(Lci_allPhasesConcatLinRegress_allDelays, delays, 'LeftCircles', downsampled_bucket_size_ms, Lci_rvalVsDelay_folder) 
# by phase
allRcoPhases = [Rco_calibLinRegress_allDelays, Rco_octoLinRegress_allDelays, Rco_u1LinRegress_allDelays, Rco_u2LinRegress_allDelays, Rco_u3LinRegress_allDelays, Rco_u4LinRegress_allDelays, Rco_u5LinRegress_allDelays, Rco_u6LinRegress_allDelays]
drawFitScoresVsDelay_byPhase(allRcoPhases, delays, phase_names, 'RightContours', downsampled_bucket_size_ms, Rco_rvalVsDelay_folder)
allRciPhases = [Rci_calibLinRegress_allDelays, Rci_octoLinRegress_allDelays, Rci_u1LinRegress_allDelays, Rci_u2LinRegress_allDelays, Rci_u3LinRegress_allDelays, Rci_u4LinRegress_allDelays, Rci_u5LinRegress_allDelays, Rci_u6LinRegress_allDelays]
drawFitScoresVsDelay_byPhase(allRciPhases, delays, phase_names, 'RightCircles', downsampled_bucket_size_ms, Rci_rvalVsDelay_folder)
allLcoPhases = [Lco_calibLinRegress_allDelays, Lco_octoLinRegress_allDelays, Lco_u1LinRegress_allDelays, Lco_u2LinRegress_allDelays, Lco_u3LinRegress_allDelays, Lco_u4LinRegress_allDelays, Lco_u5LinRegress_allDelays, Lco_u6LinRegress_allDelays]
drawFitScoresVsDelay_byPhase(allLcoPhases, delays, phase_names, 'LeftContours', downsampled_bucket_size_ms, Lco_rvalVsDelay_folder)
allLciPhases = [Lci_calibLinRegress_allDelays, Lci_octoLinRegress_allDelays, Lci_u1LinRegress_allDelays, Lci_u2LinRegress_allDelays, Lci_u3LinRegress_allDelays, Lci_u4LinRegress_allDelays, Lci_u5LinRegress_allDelays, Lci_u6LinRegress_allDelays]
drawFitScoresVsDelay_byPhase(allLciPhases, delays, phase_names, 'LeftCircles', downsampled_bucket_size_ms, Lci_rvalVsDelay_folder)

###################################
# save linear regression parameters as binary files
###################################
# output path
Rco_allPhasesConcat_linRegress_output = pupilSizeVsDelayLinRegress_folder + os.sep + 'pupilSizeVsDelayLinRegressParams_RightContours_allPhasesConcat_%dTBDelays.npy'%(delays)
Rci_allPhasesConcat_linRegress_output = pupilSizeVsDelayLinRegress_folder + os.sep + 'pupilSizeVsDelayLinRegressParams_RightCircles_allPhasesConcat_%dTBDelays.npy'%(delays)
Lco_allPhasesConcat_linRegress_output = pupilSizeVsDelayLinRegress_folder + os.sep + 'pupilSizeVsDelayLinRegressParams_LeftContours_allPhasesConcat_%dTBDelays.npy'%(delays)
Lci_allPhasesConcat_linRegress_output = pupilSizeVsDelayLinRegress_folder + os.sep + 'pupilSizeVsDelayLinRegressParams_LeftCircles_allPhasesConcat_%dTBDelays.npy'%(delays)
# save file
np.save(Rco_allPhasesConcat_linRegress_output, Rco_allPhasesConcatLinRegress_allDelays)
np.save(Rci_allPhasesConcat_linRegress_output, Rci_allPhasesConcatLinRegress_allDelays)
np.save(Lco_allPhasesConcat_linRegress_output, Lco_allPhasesConcatLinRegress_allDelays)
np.save(Lci_allPhasesConcat_linRegress_output, Lci_allPhasesConcatLinRegress_allDelays)

# FIN