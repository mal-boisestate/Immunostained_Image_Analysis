import os
import time
import csv
from objects.Structures import UnetParam
from objects.Analyzer import Analyzer
from objects.BioformatReader import BioformatReader
import javabridge
import bioformats
import pandas as pd
import openpyxl

# TODO: Construct this test script, decide what functionalities it should test, and determine what other components (
#  unet model, imgs, etc.) are needed

# BRAINSTORMING:

# 1) Stain analysis for still img
#   - Nucleus contour drawing in verification imgs (confirm accuracy)
#       - 20x and 63x model testing on respective imgs (and others, if added)
#   - In nuclei/in entire img/outside nuclei stain quantification across channels (confirm accuracy/functionality)
#       - Test with imgs with different numbers of channels (make sure not limited to 3)
#   - Regular area/perinuclear area (confirm accuracy/functionality)
#   - Other stat file data (confirm accuracy)
#       - Frame # (should be 0 for still imgs)
#       - File names
#       - Cell ID #s
#       - Cell location coordinates
#       - Perimeters
#       - Areas
#       - Etc.
#   - Avg stat file data (confirm accuracy)
#       - File names
#       - Cell count
#       - Avg stain quantities (total stain / # of cells)

# 2) Stain analysis for timelapse (multiple frames)
#   - Same as for still img (confirm functionalities work with timelapses)
#   - Cell movement tracking (confirm accuracy/functionality)
#   - Cell count over time (check accuracy/functionality)

# Necessary external files:
#   - all relevant unet models (think I still need 20x from Nina?)
#   - still imgs for each magnification size (1 img per magnification - to simplify statistical comparison?)
#   - 1 timelapse file (no need to test across all magnifications?)
#   - separate folder to hold temporary test analytics? (separate from normal analysis_data folder)

# Testing mechanism:
#   - Create and run analyzers with different inputs to test all parameters mentioned above
#   - For statistical data, compare test run results with pre-made results and return Pass/Fail + margin of error
#   - For visual data (nuc verification, etc.) provide pre-made visuals for manual user comparison?
#       - Better way to do this?
#   - Automatically run with each push to GitHub - Nina knows how to do this?

# TESTS:
# 1) unet, 63x img - confirms unet functionality for 63x
# 2) thr, 63x img - confirms thr functionality
# 3) unet, 63x img, perinuclear area - confirms perinuclear area functionality
# 4) unet, 63x img, cell separation - confirms cell separation functionality
# 6) unet, 20x img + model - confirms unet functionality for 20x unet model
# 5) unet recognition, 63x 2-channel img - confirms functionality with different channel numbers
#   - if more unet models are added, will have to test them too
# 7) timelapse, unet - confirms timelapse analysis functionality
# 8) timelapse, unet, track movement - confirms movement tracking functionality in timelapse

def stats_difference_analysis(df1, df2):
    """
            Takes two DataFrame objects of equal size and with equal data types, and finds the percent difference
            between the aggregated values of each.

            df1: first DataFrame; should be the control
            df2: second DataFrame; should be from program analysis

            """

    difference = df1.subtract(df2)

    percent_matrix = difference.divide(df1) * 100


    percent_sum = percent_matrix.to_numpy().sum()
    difference_size = difference.size
    percent_difference = percent_sum / difference_size

    return percent_difference, percent_matrix, difference

def signal_quantification_test(df1, df2, test_num, channel_num):
    pass_fail = None
    avg_difference = 0
    additional_notes = []

    cell_count_control = df1.iloc[0, 2]
    cell_count_test = df2.iloc[0, 2]

    # If the number of identified nuclei is equal, proceed with comparison. If not, can't do direct comparison
    if cell_count_control != cell_count_test:
        notif = "Different numbers of nuclei identified in each img"
        additional_notes.append(notif)
        print(notif)
        print("")
    else:
        print("Equal numbers of nuclei identified in each img")
        print("")

    # For avg analysis - Compare the avg signal densities between the control and experimental imgs
    avg_signal_1 = df1.iloc[0, 3:(channel_num + 3)]
    avg_signal_2 = df2.iloc[0, 3:(channel_num + 3)]

    # Find the percent difference in avg signal densities
    avg_signal_difference, signal_difference_matrix, difference = stats_difference_analysis(avg_signal_1, avg_signal_2)
    avg_difference = avg_signal_difference

    # TODO: See if I can print DataFrame of differences
    print("Percent difference between control and test results for each channel:")
    print("")
    print(signal_difference_matrix)

    print("")
    print("Combined percent difference between control and test results:")
    print(str(avg_signal_difference) + " %")
    print("")

    if avg_signal_difference != 0:
        print("Signal values aren't exactly the same")
        print("")
        additional_notes.append("Signal values aren't exactly the same")

    if abs(avg_signal_difference) <= 10:  # Currently using a 10% error cutoff
        print("Test " + str(test_num) + ": PASS - difference is less than 10%")
        pass_fail = "PASS"
    else:
        print("Test " + str(test_num) + ": FAIL - difference is greater than 10%")
        pass_fail = "FAIL"

    print("")

    # For debugging additional_notes
    additional_notes.append("Notes Test 1")
    additional_notes.append("Notes Test 2")

    return pass_fail, avg_difference, additional_notes

# TODO
def save_test_summary(test_nums, test_outcomes, test_differences, summary_output_folder, analyzer_runtime, additional_notes):
    header_row = ["Test #", "Result", "% Difference", "Analyzer Runtime", None, "Additional Notes"]

    path = os.path.join(summary_output_folder, 'tests_summary.csv')
    with open(path, mode='w', newline='') as stat_file:
        csv_writer = csv.writer(stat_file, delimiter=',')
        csv_writer.writerow(header_row)

        for i in range(0, len(test_nums)):
            csv_writer.writerow([test_nums[i], test_outcomes[i], test_differences[i], analyzer_runtime[i], None] + [note for note in additional_notes[i]])

    print("csv test summary created")

def main():

    # Initializing relevant analysis variables

    nuc_recognition_mode = "unet"
    mask_channel_name = "DAPI"
    isWatershed = False
    trackMovement = False
    trackEachFrame = False
    perinuclearArea = False
    isTimelapse = False  # necessary placeholder for analyzer constructor

    # Initializing unet characteristics

    unet_model_63x = r"unet\models\CP_epoch198.pth"
    unet_model_20x = r"unet\models\CP_epoch65_only20x_no-aug.pth"
    unet_model_scale = 1
    unet_img_size = (512, 512)
    unet_model_thrh = 0.5
    nuc_area_min_pixels_num = 200  # Minimum pixel size of contiguous ROIs to be labeled as "cells"
    unet_parm = UnetParam(unet_model_63x, unet_model_20x, unet_model_scale, unet_model_thrh, unet_img_size)
    nuc_threshold = 50  # None by default
    javabridge.start_vm(class_path=bioformats.JARS)

# Initializing input and output folders

    # For test outcomes
    summary_output_folder = r"C:\BioLab2\Immunostained_Image_Analysis\test_results\test_summary"

    # For Test 1 save folder + 63x img sample locations
    bioformat_imgs_path_63x = r"C:\BioLab2\Immunostained_Image_Analysis\test_img_63x"
    analysis_out_path_t1 = r"C:\BioLab2\Immunostained_Image_Analysis\test_results\tests\test_1"
    # For Test 2 save folder location
    analysis_out_path_t2 = r"C:\BioLab2\Immunostained_Image_Analysis\test_results\tests\test_2"
    # For Test 3 save folder location
    analysis_out_path_t3 = r"C:\BioLab2\Immunostained_Image_Analysis\test_results\tests\test_3"

    # TODO: Figure out how to make these folders reachable across devices (config?)

    # Initializing universal variables for test outcomes and percent differences
    test_counter = 1
    test_nums = [] # array counting each test using test_counter
    test_outcomes = [] # array containing the PASS/FAIL outcome for each test
    test_differences = [] # array containing percent discrepencies, where relevant
    analyzer_runtimes = [] # array containing runtimes for each test analyzer
    additional_notes_main = [] # For any notes of interest and/or concenrs that may be worth looking at

    # Starting the clock
    start = time.time()

    # Analyzer Setup for Test 1
    print("")
    print("Running Test 1")
    print("Please wait...")
    print("")

    reader1 = BioformatReader(bioformat_imgs_path_63x, 0,
                              mask_channel_name)  # for channel num; fine for almost every test
    analyser1 = Analyzer(bioformat_imgs_path_63x, nuc_recognition_mode, nuc_threshold, unet_parm,
                         nuc_area_min_pixels_num,
                         mask_channel_name, isWatershed, trackMovement, trackEachFrame, isTimelapse, perinuclearArea,
                         analysis_out_path_t1)
    analyser1.run_analysis()
    end = time.time()
    analyzer_runtimes.append(end - start)

    time_temp = end

    # Analyzer Setup for Test 2
    print("")
    print("Running Test 2")
    print("Please wait...")
    print("")

    analyser2 = Analyzer(bioformat_imgs_path_63x, "thr", nuc_threshold, unet_parm,
                         nuc_area_min_pixels_num,
                         mask_channel_name, isWatershed, trackMovement, trackEachFrame, isTimelapse, perinuclearArea,
                         analysis_out_path_t2)
    analyser2.run_analysis()
    end = time.time()
    analyzer_runtimes.append(end - time_temp)

    time_temp = end

    # Analyzer Setup for Test 3
    print("")
    print("Running Test 3")
    print("Please wait...")
    print("")

    analyser3 = Analyzer(bioformat_imgs_path_63x, "unet", nuc_threshold, unet_parm,
                         nuc_area_min_pixels_num,
                         mask_channel_name, isWatershed, trackMovement, trackEachFrame, isTimelapse, True,
                         analysis_out_path_t3)
    analyser3.run_analysis()
    end = time.time()
    analyzer_runtimes.append(end - time_temp)

    time_temp = end

    # TEST ANALYSES

    print("")
    print("TEST RESULTS:")
    print("")

    # Run Test 1
    print("----------------------------------------------------------------------------------------------------------")
    print("TEST 1 - unet, 63x img")
    print("----------------------------------------------------------------------------------------------------------")
    print("")

    df1 = pd.read_excel(r'C:\BioLab2\Immunostained_Image_Analysis\test_control_data\63x_normal.xlsx') # path to control excel file
    df2 = pd.read_excel(r'C:\BioLab2\Immunostained_Image_Analysis\test_results\tests\test_1\analysis_data\general_stats\signal_avg_xlsx.xlsx') # path to tested excel file
    # TODO: include pip install openpyxl as part of required installations?

    pass_fail, avg_difference, additional_notes = signal_quantification_test(df1, df2, test_counter, reader1.channel_nums)

    test_nums.append(test_counter)
    test_counter += 1
    test_outcomes.append(pass_fail)
    test_differences.append(avg_difference)
    additional_notes_main.append(additional_notes)

    # Run Test 2
    print("----------------------------------------------------------------------------------------------------------")
    print("TEST 2 - thr, 63x img")
    print("----------------------------------------------------------------------------------------------------------")
    print("")

    df1 = pd.read_excel(
        r'C:\BioLab2\Immunostained_Image_Analysis\test_control_data\63x_normal.xlsx')  # path to control excel file
    df2 = pd.read_excel(
        r'C:\BioLab2\Immunostained_Image_Analysis\test_results\tests\test_2\analysis_data\general_stats\signal_avg_xlsx.xlsx') # path to tested excel file
    # TODO: Adjust df1 and df2 to refer to dynamic folders - look into os.path.join or config?

    pass_fail, avg_difference, additional_notes = signal_quantification_test(df1, df2, test_counter, reader1.channel_nums)

    test_nums.append(test_counter)
    test_counter += 1
    test_outcomes.append(pass_fail)
    test_differences.append(avg_difference)
    additional_notes_main.append(additional_notes)

    # Run Test 3
    print("----------------------------------------------------------------------------------------------------------")
    print("TEST 3 - unet, 63x img, perinuclear area analysis")
    print("----------------------------------------------------------------------------------------------------------")
    print("")

    df1 = pd.read_excel(
        r'C:\BioLab2\Immunostained_Image_Analysis\test_control_data\63x_normal.xlsx')  # path to control excel file
    df2 = pd.read_excel(
        r'C:\BioLab2\Immunostained_Image_Analysis\test_results\tests\test_3\analysis_data\general_stats\signal_avg_xlsx.xlsx')  # path to tested excel file

    pass_fail, avg_difference, additional_notes = signal_quantification_test(df1, df2, test_counter,
                                                                             reader1.channel_nums)

    test_nums.append(test_counter)
    test_counter += 1
    test_outcomes.append(pass_fail)
    test_differences.append(avg_difference)
    additional_notes_main.append(additional_notes)

    # Create test summary csv compiling general test results
    save_test_summary(test_nums, test_outcomes, test_differences, summary_output_folder, analyzer_runtimes, additional_notes_main)

    print("----------------------------------------------------------------------------------------------------------")
    print("TESTING COMPLETE")
    print("----------------------------------------------------------------------------------------------------------")
    print("")

    javabridge.kill_vm()


if __name__ == '__main__':
    main()