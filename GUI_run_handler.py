import time
from objects.Structures import UnetParam
from objects.Analyzer import Analyzer
import javabridge
import bioformats


def run_through_gui(analysis_type, bioformat_imgs_path,
                    nuc_recognition_mode, mask_channel_name,nuc_area_min_pixels_num,
                    nuc_threshold, isWatershed, analysis_out_path):

    trackMovement = True if analysis_type == 'tracing' else False

    unet_model = r"unet\models\CP_epoch198.pth"  # path to the trained Unet model if the user chooses nuc_recognition_mode = unet if not can be None

    # Unet training process characteristics:
    unet_model_scale = 1
    unet_img_size = (512, 512)
    unet_model_thrh = 0.5
    unet_parm = UnetParam(unet_model, unet_model_scale, unet_model_thrh, unet_img_size)
    javabridge.start_vm(class_path=bioformats.JARS)

    start = time.time()
    analyser = Analyzer(bioformat_imgs_path, nuc_recognition_mode, nuc_threshold, unet_parm, nuc_area_min_pixels_num,
                        mask_channel_name, isWatershed, trackMovement, trackEachFrame=False, analysis_out_path=analysis_out_path)
    analyser.run_analysis()
    end = time.time()
    print("Total time is: ")
    print(end - start)
    javabridge.kill_vm()

