import time
from objects.Structures import UnetParam
from objects.Analyzer import Analyzer
import javabridge
import bioformats


def main():
    bioformat_imgs_path = r"D:\BioLab\img\Chase_img\20x\Replicate 3 (6-7-22)\Control"  # path to the folder that contains bio format images (czi, lif, ect) or path to the specific image
    nuc_recognition_mode = "unet"  # "unet" or "thr"
    mask_channel_name = "DAPI"
    isWatershed = False # applies watershed to separate touching cells
    trackMovement = False # toggles cell movement tracking functionality
    trackEachFrame = False # will create and save a plot of cell movement for each

    # Failsafe conditional(s) if things are missed above
    if trackMovement is False:
        trackEachFrame = False

    # unet_model = r"D:\BioLab\src_matlab_alternative\unet\models\CP_epoch65_only20x_no-aug.pth"
    # # unet_model = r"D:\BioLab\src_matlab_alternative\checkpoints\CP_epoch71.pth"
    unet_model = r"D:\BioLab\src_matlab_alternative\checkpoints\CP_epoch18.pth"  # path to the trained Unet model if the user chooses nuc_recognition_mode = unet if not can be None

    # Unet training process characteristics:
    unet_model_scale = 1
    unet_img_size = (512, 512)
    unet_model_thrh = 0.5
    nuc_area_min_pixels_num = 200 # Minimum pixel size of contiguous ROIs to be labeled as "cells"
    unet_parm = UnetParam(unet_model, unet_model_scale, unet_model_thrh, unet_img_size)
    nuc_threshold = 120 # None by default
    javabridge.start_vm(class_path=bioformats.JARS)

    start = time.time()
    analyser = Analyzer(bioformat_imgs_path, nuc_recognition_mode, nuc_threshold, unet_parm, nuc_area_min_pixels_num,
                        mask_channel_name, isWatershed, trackMovement, trackEachFrame)
    analyser.run_analysis()
    end = time.time()
    print("Total time is: ")
    print(end - start)
    javabridge.kill_vm()



if __name__ == '__main__':
    main()
