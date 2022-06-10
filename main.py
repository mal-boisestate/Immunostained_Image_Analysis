import time
from objects.Structures import UnetParam
from objects.Analyzer import Analyzer
import javabridge
import bioformats


def main():
    bioformat_imgs_path = r"D:\BioLab\img\Images for matlab quant\10x"  # path to the folder that contains bio format images (czi, lif, ect) or path to the specific image
    nuc_recognition_mode = "unet"  # "unet" or "trh" TODO Implement threshold option. Now only unet mode work
    mask_channel_name = "DAPI"

    unet_model = r"unet\models\CP_epoch198.pth"  # path to the trained Unet model if the user chooses nuc_recognition_mode = unet if not can be None

    # Unet training process characteristics:
    unet_model_scale = 1
    unet_img_size = (512, 512)
    unet_model_thrh = 0.5
    nuc_area_min_pixels_num = 200
    unet_parm = UnetParam(unet_model, unet_model_scale, unet_model_thrh, unet_img_size)
    nuc_theshold = 30
    javabridge.start_vm(class_path=bioformats.JARS)

    start = time.time()
    analyser = Analyzer(bioformat_imgs_path, nuc_recognition_mode, nuc_theshold, unet_parm, nuc_area_min_pixels_num,
                        mask_channel_name)
    analyser.run_analysis()
    end = time.time()
    print("Total time is: ")
    print(end - start)
    javabridge.kill_vm()


if __name__ == '__main__':
    main()
