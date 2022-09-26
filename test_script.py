import time
from objects.Structures import UnetParam
from objects.Analyzer import Analyzer
import javabridge
import bioformats

# TODO: Construct this test script, decide what functionalities it should test, and determine what other components (
#  unet model, imgs, etc.) are needed

# Rought Draft:

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
#   - still imgs for each magnification size (3 imgs per magnification?)
#   - 1 timelapse file (no need to test across all magnifications?)
#   - separate folder to hold temporary test analytics? (separate from normal analysis_data folder)

# Testing mechanism:
#   - Create and run analyzers with different inputs to test all parameters mentioned above
#   - For statistical data, compare test run results with pre-made results and return Pass/Fail + margin of error
#   - For visual data (nuc verification, etc.) provide pre-made visuals for manual user comparison?
#       - Better way to do this?
#   - Automatically run with each push to GitHub - Nina knows how to do this?

def main():
    bioformat_imgs_path = r"C:\BioLab\img\testing ground"
    nuc_recognition_mode = "unet"
    mask_channel_name = "DAPI"
    isWatershed = False
    trackMovement = False
    trackEachFrame = False
    perinuclearArea = False

    unet_model_63x = r"C:\BioLab2\Immunostained_Image_Analysis\unet\models\CP_epoch198.pth"
    unet_model_20x = r"D:\BioLab\src_matlab_alternative\unet\models\CP_epoch65_only20x_no-aug.pth"
    unet_model_scale = 1
    unet_img_size = (512, 512)
    unet_model_thrh = 0.5
    nuc_area_min_pixels_num = 200  # Minimum pixel size of contiguous ROIs to be labeled as "cells"
    unet_parm = UnetParam(unet_model_63x, unet_model_20x, unet_model_scale, unet_model_thrh, unet_img_size)
    nuc_threshold = 120  # None by default
    javabridge.start_vm(class_path=bioformats.JARS)

    start = time.time()

    # First Test - All Toggles Off
    analyser = Analyzer(bioformat_imgs_path, nuc_recognition_mode, nuc_threshold, unet_parm, nuc_area_min_pixels_num,
                        mask_channel_name, isWatershed, trackMovement, trackEachFrame, perinuclearArea)
    analyser.run_analysis()
    end = time.time()
    print("Total time is: ")
    print(end - start)
    javabridge.kill_vm()


if __name__ == '__main__':
    main()