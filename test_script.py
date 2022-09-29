import time
from objects.Structures import UnetParam
from objects.Analyzer import Analyzer
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
# 3) unet, 20x img + model - confirms unet functionality for 20x unet model
#   - if more unet models are added, will have to test them too
# 4) unet, 63x img, cell separation - confirms cell separation functionality
# 5) unet, 63x img, perinuclear area - confirms perinuclear area functionality
# 6) unet recognition, 63x 2-channel img - confirms functionality with different channel numbers
# 7) timelapse, unet - confirms timelapse analysis functionality
# 8) timelapse, unet, track movement - confirms movement tracking functionality in timelapse

def signal_quantification_test(df1, df2, test_num):
    cell_count_control = df1.loc[:, "Cell id, #"]
    cell_count_test = df2.loc[:, "Cell id, #"]

    # If the number of identified nuclei is equal, proceed with comparison. If not, can't do direct comparison
    if cell_count_control.size != cell_count_test.size:
        print("Analysis identified different numbers of nuclei in each img - direct comparison impossible")
        print("Test " + str(test_num) + ": FAIL - direct comparison impossible")
        print("")
    else:
        signal_data_1 = df1.iloc[:, 4:11]
        signal_data_2 = df2.iloc[:, 4:11]

        difference = signal_data_1.subtract(signal_data_2)

        percent_difference = difference.divide(signal_data_1) * 100

        percent_difference_sum = percent_difference.to_numpy().sum()
        percent_difference_size = percent_difference.size
        avg_difference = percent_difference_sum / percent_difference_size

        print("Test " + str(test_num) + " - unet recognition, 63x imgs")
        print("Percent difference between control and test results for immunofluorescent signals:")
        print(str(avg_difference) + " %")
        print("")

        if abs(avg_difference) <= 5:  # Currently using a
            print("Test " + str(test_num) + ": PASS - difference is less than 5%")
        else:
            print("Test " + str(test_num) + ": FAIL - difference is greater than 5%")

        print("Please wait...")


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

    unet_model_63x = r"C:\BioLab2\Immunostained_Image_Analysis\unet\models\CP_epoch198.pth"
    unet_model_20x = r"D:\BioLab\src_matlab_alternative\unet\models\CP_epoch65_only20x_no-aug.pth"
    unet_model_scale = 1
    unet_img_size = (512, 512)
    unet_model_thrh = 0.5
    nuc_area_min_pixels_num = 200  # Minimum pixel size of contiguous ROIs to be labeled as "cells"
    unet_parm = UnetParam(unet_model_63x, unet_model_20x, unet_model_scale, unet_model_thrh, unet_img_size)
    nuc_threshold = 120  # None by default
    javabridge.start_vm(class_path=bioformats.JARS)

    # Initializing input and output folders

    # For Test 1
    bioformat_imgs_path_60x = r"C:\BioLab2\Immunostained_Image_Analysis\test_controls\test_imgs\60x"
    analysis_out_path_t1 = r"C:\BioLab2\Immunostained_Image_Analysis\test_results\tests\test_1"

    # For Test 2
    analysis_out_path_t2 = r"C:\BioLab2\Immunostained_Image_Analysis\test_results\tests\test_2"

    # TODO: Figure out how to make these folders reachable across devices (config?)

    start = time.time()

    # Setup for Test 1

    print("")
    print("1st Test - unet recognition, 63x imgs")
    print("Please wait...")
    time.sleep(5)

    analyser = Analyzer(bioformat_imgs_path_60x, nuc_recognition_mode, nuc_threshold, unet_parm, nuc_area_min_pixels_num,
                        mask_channel_name, isWatershed, trackMovement, trackEachFrame, isTimelapse, perinuclearArea,
                        analysis_out_path_t1)
    analyser.run_analysis()
    end = time.time()

    print("")
    print("Time for Test 1: ")
    print(end - start)
    print("")
    time_temp = end

    df1 = pd.read_excel(r'C:\BioLab2\Immunostained_Image_Analysis\test_controls\test_data\test_1\analysis_data\general_stats\signal_quant_xlsx.xlsx') # path to control excel file
    df2 = pd.read_excel(r'C:\BioLab2\Immunostained_Image_Analysis\test_results\tests\test_1\analysis_data\general_stats\signal_quant_xlsx.xlsx') # path to tested excel file
    # TODO: include pip install openpyxl as part of required installations?

    # Run Test 1
    signal_quantification_test(df1, df2, 1)

    time.sleep(10)

    # Setup for Test 2

    print("")
    print("2nd Test - thr recognition, 63x imgs")
    print("Please wait...")
    time.sleep(5)

    analyser2 = Analyzer(bioformat_imgs_path_60x, "thr", nuc_threshold, unet_parm,
                        nuc_area_min_pixels_num,
                        mask_channel_name, isWatershed, trackMovement, trackEachFrame, isTimelapse, perinuclearArea,
                        analysis_out_path_t2)
    analyser2.run_analysis()
    end = time.time()

    print("")
    print("Time for Test 2: ")
    print(end - time_temp)
    print("")
    time_temp = end

    df1 = pd.read_excel(
        r'C:\BioLab2\Immunostained_Image_Analysis\test_controls\test_data\test_1\analysis_data\general_stats\signal_quant_xlsx.xlsx')  # path to control excel file
    df2 = pd.read_excel(
        r'C:\BioLab2\Immunostained_Image_Analysis\test_results\tests\test_2\analysis_data\general_stats\signal_quant_xlsx.xlsx')
    # TODO: Adjust df1 and df2 to refer to dynamic folders - look into os.path.join

    # Run Test 2
    signal_quantification_test(df1, df2, 2)

    time.sleep(10)


    javabridge.kill_vm()


if __name__ == '__main__':
    main()