import numpy as np
import cv2 as cv2
import os
import math
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from objects import Contour
from objects.Structures import NucAreaData, Signal


class ImageData(object):
    def __init__(self, path, channels_raw_data, nuc_mask, nuc_area_min_pixels_num, time_point=0, isWatershed=True):
        self.path = path
        self.channels_raw_data = channels_raw_data
        self.nuc_mask = nuc_mask
        self.cnts = self._get_nuc_cnts(isWatershed, nuc_area_min_pixels_num)
        self.cells_data, self.cells_num = self._analyse_signal_in_nuc_area(nuc_area_min_pixels_num)
        self.time_point = time_point


    def _get_nuc_cnts(self, isWatershed, nuc_area_min_pixels_num):
        full_cnts = []
        if not isWatershed:
            full_cnts = Contour.get_mask_cnts(self.nuc_mask) # contours drawn from provided nuc_mask (a binary 1/255 arr)
        else: # Applying watershed algorithm on the mask
            distance = ndi.distance_transform_edt(self.nuc_mask)
            min_distance = 2 * int((nuc_area_min_pixels_num / math.pi) ** 1/2) # diameter that based on formula of Area of a circle
                                                                               # scales to the cell size threshold defined in the main()
            coords = peak_local_max(distance, min_distance=min_distance, labels=self.nuc_mask)
            mask = np.zeros(distance.shape, dtype=bool)
            mask[tuple(coords.T)] = True
            markers, _ = ndi.label(mask)
            labels = watershed(-distance, markers, mask=self.nuc_mask)

            # loops through labels and removes any cells that touch the edges of the frame
            for x in range(0, len(labels)):
                for y in range(0, len(labels)):
                    if x == len(labels) - 1 and labels[y][x] != 0 \
                            or (x == 0 and labels[y][x] != 0) or (y == len(labels) - 1 and labels[y][x] != 0) \
                            or (y == 0 and labels[y][x] != 0):
                        temp_elim = labels[y][x]
                        for a in range(0, len(labels)):
                            for b in range(0, len(labels)):
                                if labels[b][a] == temp_elim:
                                    labels[b][a] = 0

            #Find cntrs
            for label in np.unique(labels): # np.unique() finds the unique element(s) of an array
                                            # in this case, any non-0 values (labeled coordinates) will stand out as unique
            # don't need iterable element in for loops?
                if label == 0:
                    continue # "continue" statement loops back to the start of the loop, without executing rest of code
                label_mask = np.zeros_like(labels, dtype=np.uint8) # unlike in the example, data type is set to int8
                                                                   # so that no bool array is needed
                label_mask[labels == label] = 255
                full_cnts.extend(cv2.findContours(label_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0])
                # "extend" adds a specified element to the end of a given list
                # Returns a list of contours

        return full_cnts


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
            if channel.name == mask_img_name:
                cv2.imwrite(img_path, img_8bit)
            merged_img.append(img_8bit)

        if len(merged_img) == 3:
            color_img_path = os.path.join(output_folder,
                                    base_img_name + '_color' + '_t-' + str(t) +'.png')
            color_img = cv2.merge(merged_img)
            cv2.imwrite(color_img_path, color_img)
