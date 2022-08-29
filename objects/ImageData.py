import sys
import numpy as np
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt
import cv2 as cv2
import os
import math
import skimage
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from objects import Contour
from objects.Structures import NucAreaData, Signal

def run_erosion_dialation(nuc_mask):
    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(nuc_mask, kernel, iterations=1)
    img_dilation = cv2.dilate(nuc_mask, kernel, iterations=1)
    return nuc_mask


class ImageData(object):
    def __init__(self, path, channels_raw_data, nuc_mask, nuc_area_min_pixels_num, time_point=0, isWatershed=True, trackMovement=False, features=None):
        self.path = path
        self.channels_raw_data = channels_raw_data
        self.nuc_mask = run_erosion_dialation(nuc_mask) #helps to remove
        self.cnts, self.features = self._get_nuc_cnts(isWatershed, nuc_area_min_pixels_num, time_point, trackMovement, features)
        self.cells_data, self.cells_num = self._analyse_signal_in_nuc_area(nuc_area_min_pixels_num)
        self.time_point = time_point
        # self.features = self._get_features() TODO: We can add other characteristics such as intensity  and area and organize it im one function
        self.signals_list = [] # list that will contain signal intensities for each time point


    def _get_nuc_cnts(self, isWatershed, nuc_area_min_pixels_num, t=0, trackMovement=False, features=None): # add last three to ImageData object!
        # features is the DataFrame object to which cell location data will be added
        self.remove_edge_cells() #  Remove cells on the edge of image from the nucleus mask
        full_cnts = []
        cell_num = 1

        if not isWatershed:
            new_nuc_mask = self.nuc_mask
            need_increment = True
            if trackMovement is True:
                features = self.find_nuc_locations(new_nuc_mask, features, need_increment, t, cell_num, trackMovement)
            full_cnts = Contour.get_mask_cnts(new_nuc_mask) # contours drawn from provided nuc_mask (a binary 1/255 arr)

        else: # Applying watershed algorithm on the mask
            need_increment = False
            distance = ndi.distance_transform_edt(self.nuc_mask)
            min_distance = 2 * int((nuc_area_min_pixels_num / math.pi) ** 1/2) # diameter that based on formula of Area of a circle
                                                                               # scales to the cell size threshold defined in the main()
            coords = peak_local_max(distance, min_distance=min_distance, labels=self.nuc_mask)
            mask = np.zeros(distance.shape, dtype=bool)
            mask[tuple(coords.T)] = True
            markers, _ = ndi.label(mask)
            labels = watershed(-distance, markers, mask=self.nuc_mask)

            # loops through labels and removes any cells that touch the edges of the frame

            # Find cntrs
            for label in np.unique(labels): # np.unique() finds the unique element(s) of an array
                                            # in this case, any non-0 values (labeled coordinates) will stand out as unique
            # don't need iterable element in for loops?
                if label == 0:
                    continue # "continue" statement loops back to the start of the loop, without executing rest of code
                label_mask = np.zeros_like(labels, dtype=np.uint8) # unlike in the example, data type is set to int8
                                                                   # so that no bool array is needed
                label_mask[labels == label] = 255

                # there should only be 1 nucleus in each label_mask iteration
                features = self.find_nuc_locations(label_mask, features, need_increment, t, cell_num, trackMovement)
                cell_num += 1

                full_cnts.extend(cv2.findContours(label_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0])
                # "extend" adds a specified element to the end of a given list
                # Returns a list of contours

        return full_cnts, features


    def _analyse_signal_in_nuc_area(self, nuc_area_min_pixels_num):
        nuclei_area_data = []
        for cnt in self.cnts:
            mask = Contour.draw_cnt(cnt, self.nuc_mask.shape)
            center = Contour.get_cnt_center(cnt)
            area = cv2.contourArea(cnt)
            if area < nuc_area_min_pixels_num:  # if it is noise not a nuc
                continue
            nucleus_area_data = NucAreaData(center, area)
            signals = []
            for channel in self.channels_raw_data:
                cut_out_signal_img = np.multiply(mask, channel.img)
                signal_sum = np.matrix.sum(np.asmatrix(cut_out_signal_img))
                signal = Signal(channel.name, signal_sum)
                signals.append(signal)

            nucleus_area_data.update_signals(signals)
            nuclei_area_data.append(nucleus_area_data)
        return nuclei_area_data, len(nuclei_area_data)

    def draw_and_save_cnts_for_channels(self, output_folder, nuc_area_min_pixels_num, mask_img_name, t=0):
        base_img_name = os.path.splitext(os.path.basename(self.path))[0]
        cnts = [cnt for cnt in self.cnts if cv2.contourArea(cnt) > nuc_area_min_pixels_num]
        merged_img = []

        for channel in self.channels_raw_data:
            img_path = os.path.join(output_folder, base_img_name + '_' + channel.name + '_t-' + str(t) + '.png')
            img_8bit = cv2.normalize(channel.img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            cv2.drawContours(img_8bit, cnts, -1, (255, 255, 50), 3)
            for i, cnt in enumerate(cnts):
                org = Contour.get_cnt_center(cnt)
                cv2.putText(img_8bit, str(i), org, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(255, 255, 0), thickness=3)

            if channel.name == mask_img_name:
                cv2.imwrite(img_path, img_8bit)
            merged_img.append(img_8bit)

        if len(merged_img) == 3:
            color_img_path = os.path.join(output_folder,
                                    base_img_name + '_color' + '_t-' + str(t) +'.png')
            color_img = cv2.merge(merged_img)
            cv2.imwrite(color_img_path, color_img)

    def find_nuc_locations(self, nuc_mask, features, need_increment, t=0, cell_num=1, trackMovement=False, output_folder=None):

        if trackMovement is True:

            black = 0
            label_image = skimage.measure.label(nuc_mask, background=black)

            for region in skimage.measure.regionprops(label_image, intensity_image=nuc_mask):
                # Everywhere, skip small areas
                if region.area < 5:
                    continue
                # Only white areas
                if region.mean_intensity < 255:
                    continue

                # Store features which survived the above criteria
                features = features.append([{'y': region.centroid[0],
                                             'x': region.centroid[1],
                                             'cell #': cell_num,
                                             'frame': t
                                             }, ])

                if need_increment is True:
                    cell_num += 1

        return features

    def remove_edge_cells(self): # removes cells that touch the edges of the frame
        cnts = Contour.get_mask_cnts(self.nuc_mask)
        max_x, max_y = self.nuc_mask.shape

        if max_x != max_y:
            sys.exit("The current version of the program can analyze only square shape images."
                        "Please modify remove_edge_cells to overcome this issue.")

        new_cnts = [cnt for cnt in cnts if cnt.max() < max_x - 2 and cnt.min() > 1]
        nuc_mask_no_edge_cells = np.zeros(self.nuc_mask.shape, dtype="uint8")
        cv2.drawContours(nuc_mask_no_edge_cells, new_cnts, -1, color=(255, 255, 255), thickness=cv2.FILLED)
        self.nuc_mask = nuc_mask_no_edge_cells


    def new_analyse_signal_in_nuc_area(self, center, nuc_area_min_pixels_num): # NON-FUNCTIONAL - TODO
        nuclei_area_data = []
        for cnt in self.cnts:
            mask = Contour.draw_cnt(cnt, self.nuc_mask.shape)
            center = Contour.get_cnt_center(cnt)
            area = cv2.contourArea(cnt)
            if area < nuc_area_min_pixels_num:  # if it is noise not a nuc
                continue
            nucleus_area_data = NucAreaData(center, area)
            signals = []
            for channel in self.channels_raw_data:
                cut_out_signal_img = np.multiply(mask, channel.img)
                signal_sum = np.matrix.sum(np.asmatrix(cut_out_signal_img))
                signal = Signal(channel.name, signal_sum)
                signals.append(signal)

            nucleus_area_data.update_signals(signals)
            nuclei_area_data.append(nucleus_area_data)
        return nuclei_area_data, len(nuclei_area_data)