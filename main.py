import time
from objects.Structures import UnetParam
from objects.Analyzer import Analyzer
import javabridge
import bioformats


def main():
    # bioformat_imgs_path = r"D:\BioLab\img\Chase_img\63x\Replicate 1 (3-19-22)-20220908T033855Z-001\Replicate 1 (3-19-22)\6X LIV"  # path to the folder that contains bio format images (czi, lif, ect) or path to the specific image
    bioformat_imgs_path = r"C:\BioLab\img\testing ground"
    nuc_recognition_mode = "unet"  # "unet" or "thr"
    mask_channel_name = "DAPI"
    isWatershed = False # applies watershed to separate touching cells
    trackMovement = False # toggles cell movement tracking functionality
    trackEachFrame = False # will create and save a plot of cell movement for each
    perinuclearArea = False # Option to slightly dilate area analyzed per cell, to accommodate perinuclear stain

    # Failsafe conditional(s) if things are missed above
    if trackMovement is False:
        trackEachFrame = False

    # unet_model_63x = r"D:\BioLab\src_matlab_alternative\unet\models\CP_epoch198.pth" # path to the trained Unet model if the user chooses nuc_recognition_mode = unet if not can be None
    unet_model_63x = r"C:\BioLab2\Immunostained_Image_Analysis\unet\models\CP_epoch198.pth"
    unet_model_20x = r"D:\BioLab\src_matlab_alternative\unet\models\CP_epoch65_only20x_no-aug.pth"
    # Unet training process characteristics:
    unet_model_scale = 1
    unet_img_size = (512, 512)
    unet_model_thrh = 0.5
    nuc_area_min_pixels_num = 200 # Minimum pixel size of contiguous ROIs to be labeled as "cells"
    unet_parm = UnetParam(unet_model_63x, unet_model_20x, unet_model_scale, unet_model_thrh, unet_img_size)
    nuc_threshold = 120 # None by default
    javabridge.start_vm(class_path=bioformats.JARS)

    start = time.time()
    # TODO Fix the organization of variables - particularly for perinuclearArea
    analyser = Analyzer(bioformat_imgs_path, nuc_recognition_mode, nuc_threshold, unet_parm, nuc_area_min_pixels_num,
                        mask_channel_name, isWatershed, trackMovement, trackEachFrame, perinuclearArea)
    analyser.run_analysis()
    end = time.time()
    print("Total time is: ")
    print(end - start)
    javabridge.kill_vm()



if __name__ == '__main__':
    main()
