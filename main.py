import time
from objects.Structures import UnetParam
from objects.Analyser import Analyser
import javabridge
import bioformats


def main():

    bioformat_imgs_path = r"D:\BioLab\img\Images for matlab quant\10x"
    nuc_theshold = 30
    unet_model = r"D:\BioLab\src\checkpoints\CP_epoch195.pth"
    unet_model_scale = 1
    unet_img_size = (512, 512)
    unet_model_thrh = 0.5
    nuc_area_min_pixels_num = 200

    unet_parm = UnetParam(unet_model, unet_model_scale, unet_model_thrh, unet_img_size)
    nuc_recognition_mode = "unet" #"unet" or "trh" TODO Implement threshold option. Now only matlab option work
    javabridge.start_vm(class_path=bioformats.JARS)

    start = time.time()
    analyser = Analyser(bioformat_imgs_path, nuc_recognition_mode, nuc_theshold, unet_parm, nuc_area_min_pixels_num)
    analyser.run_analysis()
    end = time.time()
    print("Total time is: ")
    print(end - start)
    javabridge.kill_vm()


if __name__ == '__main__':
    main()