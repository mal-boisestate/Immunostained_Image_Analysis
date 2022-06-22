import time
from objects.Structures import UnetParam
from objects.Analyzer import Analyzer
import javabridge
import bioformats


def main():
    bioformat_imgs_path = r"D:\BioLab\img\Overnight time-lapse"  # path to the folder that contains bio format images (czi, lif, ect) or path to the specific image
    # bioformat_imgs_path = r"D:\BioLab\img\Images for matlab quant\63x_3img_test"
    nuc_recognition_mode = "unet"  # "unet" or "thr"
    mask_channel_name = "DAPI"
    analysis_type = "nuc_count" #"nuc_count" or "nuc_area_signal"
    isWatershed = True

    unet_model = r"unet\models\CP_epoch198.pth"  # path to the trained Unet model if the user chooses nuc_recognition_mode = unet if not can be None

    # Unet training process characteristics:
    unet_model_scale = 1
    unet_img_size = (512, 512)
    unet_model_thrh = 0.5
    nuc_area_min_pixels_num = 80 # Identify the difference between this and nuc_threshold?
    unet_parm = UnetParam(unet_model, unet_model_scale, unet_model_thrh, unet_img_size)
    nuc_threshold = 50 # None by default
    javabridge.start_vm(class_path=bioformats.JARS)

    start = time.time()
    analyser = Analyzer(bioformat_imgs_path, nuc_recognition_mode, analysis_type, nuc_threshold, unet_parm, nuc_area_min_pixels_num,
                        mask_channel_name, isWatershed)
    analyser.run_analysis()
    end = time.time()
    print("Total time is: ")
    print(end - start)
    javabridge.kill_vm()


if __name__ == '__main__':
    main()
