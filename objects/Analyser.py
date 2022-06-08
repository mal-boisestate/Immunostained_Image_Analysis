import os
import numpy as np
from objects.BioformatReader import BioformatReader
from objects import Utils
from objects.ImageData import ImageData
import cv2.cv2 as cv2
import sys
import csv
import math


AREA_TH = 2000
from unet.predict import run_predict_unet

temp_folders = {
    "cut_8bit_img": 'temp/cut_img_for_unet',
    "cut_mask": 'temp/cut_mask',
    "nucleus_top_mask": 'temp/nucleus_top_mask'
}

analysis_data_folders = {
    "analysis" : 'analysis_data/stat',
    "all_layers": 'analysis_data/raw_imgs',
    "cnts_verification": 'analysis_data/nuclei_area_verification'
}


class Analyser(object):
    def __init__(self, bioformat_imgs_path, nuc_recognition_mode, nuc_theshold=None, unet_parm=None, nuc_area_min_pixels_num=0):
        self.imgs_path = bioformat_imgs_path
        self.nuc_recognition_mode = nuc_recognition_mode
        self.nuc_theshold = nuc_theshold
        self.unet_parm = unet_parm
        self.nuc_area_min_pixels_num = nuc_area_min_pixels_num

    def _save_stat(self, imgs_data, output_folder):
        #1. Check that channal names for all imagies the same
        channels_names = [channel.name for channel in imgs_data[0].channels_raw_data]
        for img_data in imgs_data:
            for i, name in enumerate(channels_names):
                if img_data.channels_raw_data[i].name != name:
                    print("Images cannot be analyzed."
                          "Channels order are not in the same for all images")
                    sys.exit()

        #2.Create column names
        header_row = ["Image name", "Cell id, #", "Cell center coordinates, (x, y)",
                      "Nucleus area, pixels" ] + [name + ', intensity' for name in channels_names]

        #3. Write data
        path = os.path.join(output_folder, 'stat.csv')
        with open(path, mode='w') as stat_file:
            csv_writer = csv.writer(stat_file, delimiter=',')
            csv_writer.writerow(header_row)
            for img_data in imgs_data:
                for i, cell in enumerate(img_data.cells_data):
                    csv_writer.writerow([img_data.path, str(i), str(cell.center), str(cell.area)] +
                                        [signal.intensity for signal in cell.signals])
        print("Stat created")


    def run_analysis(self):
        for folder in analysis_data_folders:
            Utils.prepare_folder(analysis_data_folders[folder])

        imgs_data = []

        for i, filename in enumerate(os.listdir(self.imgs_path)):
            for folder in temp_folders:
                Utils.prepare_folder(temp_folders[folder])
            reader = BioformatReader(self.imgs_path, i)
            nuc_img_8bit_norm, nuc_file_name = reader.read_nucleus_layers(norm=True)
            pieces_num = Utils.cut_image(nuc_img_8bit_norm, nuc_file_name, self.unet_parm.unet_img_size, temp_folders["cut_8bit_img"])

            run_predict_unet(temp_folders["cut_8bit_img"], temp_folders["cut_mask"],
                            self.unet_parm.unet_model_path,
                            self.unet_parm.unet_model_scale,
                            self.unet_parm.unet_model_thrh)
            nuc_mask = Utils.stitch_mask(temp_folders["cut_mask"], self.unet_parm.unet_img_size, pieces_num)
            channels_raw_data = reader.read_all_layers(analysis_data_folders["all_layers"])
            img_data = ImageData(filename, channels_raw_data, nuc_mask, self.nuc_area_min_pixels_num)
            img_data.draw_and_save_cnts_for_channels(analysis_data_folders["cnts_verification"], self.nuc_area_min_pixels_num)
            imgs_data.append(img_data)

        self._save_stat(imgs_data, analysis_data_folders["analysis"])








