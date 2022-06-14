import os
from PIL import Image, ImageFilter
from objects.BioformatReader import BioformatReader
from objects.ImageData import ImageData
import sys
import csv
import glob
import math
import cv2 as cv2
import numpy as np
from unet.predict import run_predict_unet

temp_folders = {
    "cut_8bit_img": 'temp/cut_img_for_unet',
    "cut_mask": 'temp/cut_mask'
}

analysis_data_folders = {
    "analysis": 'analysis_data/stat',
    "cnts_verification": 'analysis_data/nuclei_area_verification'
}


def make_padding(img, final_img_size):
    """
    Create padding for provided image
    :param img: image to add padding to
    :param final_img_size: tuple
    :return: padded image
    """
    h, w = img.shape[:2]
    h_out, w_out = final_img_size

    top = (h_out - h) // 2
    bottom = h_out - h - top
    left = (w_out - w) // 2
    right = w_out - w - left

    padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded_img


def cut_image(img, img_name, cut_img_size, output_folder):
    """
    Cut image (img) into smaller images of provided size (cut_img_size).
    Save small images to the specified folder with a distinctive name pattern: "img_name_y-Y_x-X.png"
    where Y and X are indexes of small cut images in the initial large image.
    Such naming needed for further image reconstruction.
    :param img: image to cut
    :param img_name: image name
    :param cut_img_size: size of final images
    :param output_folder: folder to save files to
    :return:
    """
    base_img_name = os.path.splitext(os.path.basename(img_name))[0]
    padded_img_size = (math.ceil(img.shape[0] / cut_img_size[0]) * cut_img_size[0],
                       math.ceil(img.shape[1] / cut_img_size[1]) * cut_img_size[1])

    padded_img = make_padding(img, padded_img_size)
    y_start = 0
    y_end = cut_img_size[0]
    y_order = 0
    while (padded_img_size[0] - y_end) >= 0:
        x_start = 0
        x_end = cut_img_size[1]
        x_order = 0
        while (padded_img_size[1] - x_end) >= 0:
            current_img = padded_img[y_start:y_end, x_start:x_end]
            img_path = os.path.join(output_folder,
                                    base_img_name + "_y-" + str(y_order) + '_x-' + str(x_order) + '.png')
            cv2.imwrite(img_path, current_img)
            x_start = x_end
            x_end = x_end + cut_img_size[1]
            x_order = x_order + 1
        y_start = y_end
        y_end = y_end + cut_img_size[0]
        y_order = y_order + 1
    return math.ceil(img.shape[0] / cut_img_size[0])


def prepare_folder(folder):
    """
    Create folder if it has not been created before
    or clean the folder
    ---
    Parameters:
    -   folder (string): folder's path
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    for f in glob.glob(folder + "/*"):
        os.remove(f)


def save_stat(imgs_data):
    """
    Extract and save statistical data
    :param imgs_data: an object that has image info
    """
    # 1. Check that channal names for all imagies the same
    channels_names = [channel.name for channel in imgs_data[0].channels_raw_data]
    for img_data in imgs_data:
        for i, name in enumerate(channels_names):
            if img_data.channels_raw_data[i].name != name:
                print("Images cannot be analyzed."
                      "Channels are not in the same order for all images")
                sys.exit()

    # 2.Create column names
    header_row = ["Image name", "Cell id, #", "Cell center coordinates, (x, y)",
                  "Nucleus area, pixels"] + [name + ', intensity' for name in channels_names]

    # 3. Write data
    path = os.path.join(analysis_data_folders["analysis"], 'stat.csv')
    with open(path, mode='w') as stat_file:
        csv_writer = csv.writer(stat_file, delimiter=',')
        csv_writer.writerow(header_row)
        for img_data in imgs_data:
            for i, cell in enumerate(img_data.cells_data):
                csv_writer.writerow([img_data.path, str(i), str(cell.center), str(cell.area)] +
                                    [signal.intensity for signal in cell.signals])
    print("Stat created")


def stitch_mask(input_folder, unet_img_size, num):
    img_col = []
    for i in range(num):
        img_row = []
        for img_path in glob.glob(os.path.join(input_folder, f"*_y-{i}_*.png")):
            nucleus_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_row.append(nucleus_img)
        img_col.append(cv2.hconcat(img_row))
    stitched_img = cv2.vconcat(img_col)
    # img_no_padding = remove_padding(stitched_img) #TODO this function shuld be created in case if img was padded before
    # cv2.imshow("stitched_mask", cv2.resize(stitched_img, (750, 750)))  # keep it for debugging
    # cv2.waitKey()
    return stitched_img


class Analyzer(object):
    def __init__(self, bioformat_imgs_path, nuc_recognition_mode, nuc_threshold=None, unet_parm=None,
                 nuc_area_min_pixels_num=0, mask_channel_name="DAPI"):
        self.imgs_path = bioformat_imgs_path
        self.nuc_recognition_mode = nuc_recognition_mode
        self.nuc_threshold = nuc_threshold
        self.unet_parm = unet_parm
        self.nuc_area_min_pixels_num = nuc_area_min_pixels_num
        self.mask_channel_name = mask_channel_name

    def run_analysis(self):
        for folder in analysis_data_folders:
            prepare_folder(analysis_data_folders[folder])  # prepares analysis data folder

        imgs_data = []
        for i, filename in enumerate(os.listdir(self.imgs_path)):
            for folder in temp_folders:
                prepare_folder(temp_folders[folder])
            reader = BioformatReader(self.imgs_path, i, self.mask_channel_name) # "reader" is a BioformatReader object,
                                                                                # containing all the information of the
                                                                                # image type

            # The above for loop (for i) runs through all of this code for each individual image in the folder
            # Thus, the information in reader is specific to each individual image

            if self.nuc_recognition_mode == 'unet':
                nuc_mask = self.find_mask_based_on_unet(reader)

            elif self.nuc_recognition_mode == 'thr':
                # find_mask_based_on_thr doesn't have functionality yet
                nuc_mask = self.find_mask_based_on_thr(reader, i)
            else:
                print("The recognition mode is not specified or specified incorrectly. Please use \"unet\" or \"thr\"")
                sys.exit()

            channels_raw_data = reader.read_all_layers()
            img_data = ImageData(filename, channels_raw_data, nuc_mask, self.nuc_area_min_pixels_num)
            img_data.draw_and_save_cnts_for_channels(analysis_data_folders["cnts_verification"],
                                                     self.nuc_area_min_pixels_num)
            imgs_data.append(img_data)

        save_stat(imgs_data)

    # TODO: implement this function. The algorithm shuld be simular to finding mask in MatLab program. Apply filtering(noise redution) and theshold provided by user.
    def find_mask_based_on_thr(self, reader, img_number):
        # use self.nuc_threshold
        # Look at self._remove_small_particles function it can be helpful - might not need though?
        # Logic will be noise reduction (Gaussian filter) -> thresholding (0-100 scale) -> returning nuc_mask

        # img_path is a string, img_number comes from run_analysis loop
        img_path, img_series = reader.image_path, reader.image_series  # What's the deal with image "series"?

        img = Image.open(img_path)

        # Should produce Gaussian blurred version of image
        img = img.filter(ImageFilter.GaussianBlur)

        # image_depth = reader.depth()

        # gauss_nuc_img = nuc_img.filter(ImageFilter.GaussianBlur)

        nuc_mask = 1
        return nuc_mask

    def find_mask_based_on_unet(self, reader):
        """
        Finds mask picture based on unet model. Since my GPU can handle only 512*512 images for prediction
        :param reader:
        :return:
        """
        nuc_img_8bit_norm, nuc_file_name = reader.read_nucleus_layers(norm=True)
        pieces_num = cut_image(nuc_img_8bit_norm, nuc_file_name, self.unet_parm.unet_img_size,
                               temp_folders["cut_8bit_img"])

        run_predict_unet(temp_folders["cut_8bit_img"], temp_folders["cut_mask"],
                         self.unet_parm.unet_model_path,
                         self.unet_parm.unet_model_scale,
                         self.unet_parm.unet_model_thrh)
        nuc_mask = stitch_mask(temp_folders["cut_mask"], self.unet_parm.unet_img_size, pieces_num)
        return nuc_mask

    def _remove_small_particles(self, mask):
        mask = cv2.morphologyEx(mask.astype('uint8'), cv2.MORPH_OPEN, np.ones((5, 5)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5)))
        clean_mask = np.zeros(mask.shape)
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < self.nuc_area_min_pixels_num:
                continue
            cv2.fillPoly(clean_mask, pts=[cnt], color=(255, 255, 255))
        return clean_mask.astype('uint8')
