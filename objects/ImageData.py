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
from objects.Structures import NucData, Signal

def run_erosion_dialation(nuc_mask):
    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(nuc_mask, kernel, iterations=1)
    img_dilation = cv2.dilate(nuc_mask, kernel, iterations=1)
    return nuc_mask


class ImageData(object):
    def __init__(self, path, channels_raw_data, nuc_mask, nuc_area_min_pixels_num, time_point=0, isWatershed=True, trackMovement=False, features=None, perinuclearArea=False):
        self.path = path
        self.channels_raw_data = channels_raw_data
        self.nuc_mask = run_erosion_dialation(nuc_mask) #helps to remove
        self.perinuc_mask = self.nuc_mask # make be modified later in program if perinuclear area is wanted
        self.cnts, self.perinuclear_cnts, self.features = self._get_nuc_cnts(isWatershed, nuc_area_min_pixels_num, time_point, trackMovement, features, perinuclearArea)
        self.original_cnts, _, _ = self._get_nuc_cnts(isWatershed, nuc_area_min_pixels_num, time_point, trackMovement, features, False) # always finds cnts without perinuclear area
        self.cells_data, self.cells_num, self.perinuclear_cells_data = self._analyse_signal_in_nuc_area(nuc_area_min_pixels_num, perinuclearArea)
        self.time_point = time_point
        # self.features = self._get_features() TODO: We can add other characteristics such as intensity  and area and organize it im one function
        self.signals_list = [] # list that will contain signal intensities for each time point
        self.overall_signal = self._analyze_signal_in_entire_frame() # analyzes signals across whole 2048 x 2048 img
        self.perinuclearArea = perinuclearArea # if user wants to include perinuclear area in analysis
        self.external_signal = self._analyze_signal_outside_nuclei(self.overall_signal, self.cells_data) # signals outside nuclei
        self.temp = 5


    def _get_nuc_cnts(self, isWatershed, nuc_area_min_pixels_num, t=0, trackMovement=False, features=None, perinuclearArea=False): # add last three to ImageData object!
        # features is the DataFrame object to which cell location data will be added
        self.remove_edge_cells() #  Remove cells on the edge of image from the nucleus mask
        full_cnts = []
        perinuclear_cnts = []

        original_mask = None

        cell_num = 1

        if not isWatershed: # no watershedding (no post-unet/thresholding separation of touching cells)
            if perinuclearArea is True:
                kernel = np.ones((10, 10), np.uint8)
                img_dilation = cv2.dilate(self.nuc_mask, kernel, iterations=1)
                original_mask = self.nuc_mask
                self.perinuc_mask = img_dilation # modification for use in nuc area signal analysis
                new_nuc_mask = img_dilation
            else:
                original_mask = self.nuc_mask
                new_nuc_mask = self.nuc_mask
                # original_mask used for creating original cnt layer, new_nuc_mask used for creating larger layer

            need_increment = True

            if trackMovement is True:
                features = self.find_nuc_locations(new_nuc_mask, features, need_increment, t, cell_num, trackMovement)

            full_cnts = Contour.get_mask_cnts(original_mask) # contours drawn from provided nuc_mask (a binary 1/255 arr)

            if perinuclearArea is True:
                perinuclear_cnts = Contour.get_mask_cnts(new_nuc_mask) # obtain perinuclear cnts if wanted

        # TODO: Make updated perinuclear method work with watershedding
        else: # Applying watershed algorithm on the mask
            if perinuclearArea is True:
                kernel = np.ones((10, 10), np.uint8)
                img_dilation = cv2.dilate(self.nuc_mask, kernel, iterations=1)
                self.nuc_mask = img_dilation

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

        return full_cnts, perinuclear_cnts, features


    def _analyse_signal_in_nuc_area(self, nuc_area_min_pixels_num, perinuclearArea):

        nuclei_data = []
        perinuclei_data = []
        perinuclear_counter = 0

        for cnt in self.cnts:

            save_signal_sum = []

            mask = Contour.draw_cnt(cnt, self.nuc_mask.shape)
            center = Contour.get_cnt_center(cnt)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)

            if area < nuc_area_min_pixels_num:  # if it is noise not a nuc
                continue

            nucleus_data = NucData(center, area, perimeter)
            signals = []
            for channel in self.channels_raw_data: # iterates through each existing channel
                cut_out_signal_img = np.multiply(mask, channel.img)
                signal_sum = np.matrix.sum(np.asmatrix(cut_out_signal_img))
                save_signal_sum.append(signal_sum)
                signal = Signal(channel.name, signal_sum)
                signals.append(signal)

            if perinuclearArea is True:
                perinuclear_mask = Contour.draw_cnt(self.perinuclear_cnts[perinuclear_counter], self.perinuc_mask.shape)
                perinuclear_area = cv2.contourArea(self.perinuclear_cnts[perinuclear_counter]) - area
                perinuclear_data = NucData(center, perinuclear_area, perimeter) # perimeter is inaccurate and not used

                perinuclear_counter += 1

                perinuclear_signals = []
                perinuc_channel_counter = 0
                for channel2 in self.channels_raw_data:
                    perinuc_cut_out_signal_img = np.multiply(perinuclear_mask, channel2.img)
                    perinuclear_signal_sum = np.matrix.sum(np.asmatrix(perinuc_cut_out_signal_img))
                    perinuclear_signal_true = perinuclear_signal_sum - save_signal_sum[perinuc_channel_counter]
                    perinuclear_signal = Signal(channel2.name, perinuclear_signal_true)
                    perinuclear_signals.append(perinuclear_signal)

                    perinuc_channel_counter += 1

                perinuclear_data.update_signals(perinuclear_signals)
                perinuclei_data.append(perinuclear_data)

            nucleus_data.update_signals(signals)
            nuclei_data.append(nucleus_data)

        return nuclei_data, len(nuclei_data), perinuclei_data # nuclei_data and perinuclei_data should have the same length

    def _analyze_signal_in_entire_frame(self):

        # Analyzes signal for each channel across the entire 2048 x 2048 frame

        overall_signal = []
        overall_signal_num = []

        for channel in self.channels_raw_data:

            # signal_sum = np.matrix.sum(np.asmatrix(channel.img.astype(np.int64)))
            signal_sum = np.sum(channel.img.astype(np.int64))
            signal = Signal(channel.name, signal_sum)
            overall_signal.append(signal)

        for i in range(0, len(overall_signal)):
            overall_signal_num.append(overall_signal[i].intensity)

        return overall_signal_num

    # TODO - Confirm correct functionality?
    def _analyze_signal_outside_nuclei(self, overall_signals, cells_data):

        # Calculates the difference between total image signal and signal in nuclear regions

        external_signals_channels = np.zeros((len(overall_signals)))
        cells_data_sums = np.zeros((len(overall_signals))) # for calculating total stain per channel across all cells

        for x in range(0, len(cells_data)):
            for y in range(0, len(overall_signals)):

                cells_data_sums[y] += cells_data[x].signals[y].intensity

        for i in range(0, len(overall_signals)):
            a = overall_signals[i]
            b = cells_data_sums[i]
            external_signals_channels[i] = a - b

        return external_signals_channels


    def draw_and_save_cnts_for_channels(self, output_folder, nuc_area_min_pixels_num, mask_img_name, t=0, perinuclear_area=True):
        """
            Draws contours onto 8bit versions of cell imgs for visual verification. Also handles cell numbering.
            Contours are technically drawn on imgs of every channel, but only the designated nuclear channel is saved
            to the verification img folder.

            Args:
                output_folder: str of the shorthand directory where verification imgs are saved
                nuc_area_min_pixels_num: int of the minimum acceptable nuclear cell size
                                         designated in the main/gui by the user
                mask_img_name: str of the name of the nuclear stain designated in the main/gui by the user
                t: int representing the frame of the provided img. Relevant to timelapses; will be 0 for still imgs
                perinuclear_area: boolean; if true, user wants to analyze perinuclear area
                original_contours

            """

        base_img_name = os.path.splitext(os.path.basename(self.path))[0]
        cnts = [cnt for cnt in self.cnts if cv2.contourArea(cnt) > nuc_area_min_pixels_num] # removes noise
        perinuclear_cnts = [cnt for cnt in self.perinuclear_cnts if cv2.contourArea(cnt) > nuc_area_min_pixels_num]
        merged_img = []

        for channel in self.channels_raw_data:
            img_path = os.path.join(output_folder, base_img_name + '_' + channel.name + '_t-' + str(t) + '.png')
            img_8bit = cv2.normalize(channel.img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            cv2.drawContours(img_8bit, cnts, -1, (255, 255, 255), 1)

            if perinuclear_area is True:
                cv2.drawContours(img_8bit, perinuclear_cnts, -1, (150, 0, 0), 1)
            # TODO: Determine which perinuclear verification method is more desirable

            for i, cnt in enumerate(cnts):
                org = Contour.get_cnt_center(cnt)
                cv2.putText(img_8bit, str(i), org, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(255, 255, 0), thickness=3)
                # Cell numbering?

            if mask_img_name in channel.name:
                cv2.imwrite(img_path, img_8bit)
            merged_img.append(img_8bit)

        if len(merged_img) == 3: # creates a colored version of the contour img if there are exactly 3 channels
            color_img_path = os.path.join(output_folder,
                                    base_img_name + '_color' + '_t-' + str(t) +'.png')
            color_img = cv2.merge(merged_img)
            cv2.imwrite(color_img_path, color_img)

        if perinuclear_area is True: # will produce an additional verification img without dilation

            for channel in self.channels_raw_data:
                img_path = os.path.join(output_folder, base_img_name + '_non_perinuclear_' + channel.name + '_t-' + str(t) + '.png')
                img_8bit = cv2.normalize(channel.img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                         dtype=cv2.CV_8UC1)
                cv2.drawContours(img_8bit, perinuclear_cnts, -1, (255, 255, 50), 3)
                for i, cnt in enumerate(perinuclear_cnts):
                    org = Contour.get_cnt_center(cnt)
                    cv2.putText(img_8bit, str(i), org, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3,
                                color=(255, 255, 0), thickness=3)
                    # Cell numbering?

                if channel.name == mask_img_name:
                    cv2.imwrite(img_path, img_8bit)

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

        # if max_x != max_y:
        #     sys.exit("The current version of the program can analyze only square shape images."
        #                 "Please modify remove_edge_cells to overcome this issue.")

        # Countours should not touch the edges of the image
        new_cnts = []
        for cnt in cnts:
            cnt_max_x = cnt[:, 0, 0].max()
            cnt_max_y = cnt[:, 0, 1].max()
            cnt_min_x = cnt[:, 0, 0].min()
            cnt_min_y = cnt[:, 0, 1].min()
            if (cnt_max_x < max_x - 2) and (cnt_max_y < max_y - 2) and (cnt_min_x > 1) and (cnt_min_y > 1):
                new_cnts.append(cnt)

        nuc_mask_no_edge_cells = np.zeros(self.nuc_mask.shape, dtype="uint8")
        cv2.drawContours(nuc_mask_no_edge_cells, new_cnts, -1, color=(255, 255, 255), thickness=cv2.FILLED)
        self.nuc_mask = nuc_mask_no_edge_cells