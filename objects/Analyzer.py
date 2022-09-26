import os
from objects.BioformatReader import BioformatReader
from objects.ImageData import ImageData
import sys
import csv
import glob
import math
import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import trackpy as tp
from unet.predict import run_predict_unet
import pickle
import xlsxwriter

temp_folders = {
    "cut_8bit_img": 'temp/cut_img_for_unet',
    "cut_mask": 'temp/cut_mask'
}

analysis_data_folders = {
    "analysis": 'analysis_data/general_stats',
    "cnts_verification": 'analysis_data/nuclei_area_verification',
    "nuclei_count": 'analysis_data/nuclei_count',
    "movement_tracking": 'analysis_data/movement_tracking'
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

def save_stat(imgs_data, isTimelapse, analysis_out_path):
    """
    Extract and save statistical data for an image or timelapse
    :param imgs_data: an object that has image info
    :param isTimelapse: a boolean determining whether the file(s) have more than one frame
    """
    # 1. Check that channel names for all images the same
    channels_names = [channel.name for channel in imgs_data[0][0].channels_raw_data]
    for img_data in imgs_data[0]:
        for i, name in enumerate(channels_names):
            if img_data.channels_raw_data[i].name != name:
                print("Images cannot be analyzed."
                      "Channels are not in the same order for all images")
                sys.exit()

    # 2.Create column names
    header_row = ["Frame", "Image name", "Cell id, #", "Cell center coordinates, (x, y)",
                  "Nucleus area, pixels", "Nucleus perimeter, pixels"] + ['Total intensity, ' + name for name in channels_names] + \
                 ['Average intensity, ' + name for name in channels_names]

    # 3. Write data
    path = os.path.join(analysis_out_path, analysis_data_folders["analysis"], 'signal_quant_stat.csv')
    with open(path, mode='w', newline='') as stat_file:
        csv_writer = csv.writer(stat_file, delimiter=',')
        csv_writer.writerow(header_row)
        t = 0
        for img_data_t in imgs_data:
            for img_data in img_data_t:
                for i, cell in enumerate(img_data.cells_data):
                    csv_writer.writerow([t, img_data.path, str(i), str(cell.center), str(cell.area), str(cell.perimeter)] +
                                        [signal.intensity for signal in cell.signals] +
                                        [signal.intensity/cell.area for signal in cell.signals])
                if isTimelapse is True:
                    t += 1
                csv_writer.writerow([None, None, None, None, None, None] +
                                 [None for signal in cell.signals])

    # #Save data for Chase
    # header_row = ["Image name", "Number of cells", "Total nucleus area"] + \
    #              ['Total intensity, ' + name for name in channels_names]
    #
    # # 3. Write data
    # path = os.path.join(analysis_out_path, analysis_data_folders["analysis"], 'stat_for_Chase.csv')
    # with open(path, mode='w', newline='') as stat_file:
    #     csv_writer = csv.writer(stat_file, delimiter=',')
    #     csv_writer.writerow(header_row)
    #     t = 0
    #     for img_data in imgs_data:
    #         cells_num = img_data[0].cells_num
    #         total_nuc_area = 0
    #         for cell in img_data[0].cells_data:
    #             total_nuc_area += cell.area
    #
    #         csv_writer.writerow([img_data[0].path, str(cells_num), str(total_nuc_area)] +
    #                             [np.sum(channel_intecity.img) for channel_intecity in img_data[0].channels_raw_data])
    #

    print("csv stat created")

    # Conversion of csv file to xlsx file - removes original csv file

    filepath_in = path
    filepath_out = os.path.join(analysis_out_path, analysis_data_folders["analysis"], 'signal_quant_xlsx.xlsx')
    pd.read_csv(filepath_in, delimiter=",").to_excel(filepath_out, index=False)

    os.remove(filepath_in)

def save_avg_stat(imgs_data, analysis_out_path):
    """
        Extract and save average statistical data for images; Currently non-functional for timelapses
        "Average" data is calculated as the total amount of stain in nuclear regions in an image divided by the number
        of nuclei, giving average stain quantity/nucleus.

        TODO: Make sure we're on the same page about "average". Stain/nucleus vs Stain density?

        """
    # 1. Check that channel names for all images the same
    channels_names = [channel.name for channel in imgs_data[0][0].channels_raw_data]
    for img_data in imgs_data[0]:
        for i, name in enumerate(channels_names):
            if img_data.channels_raw_data[i].name != name:
                print("Images cannot be analyzed."
                      "Channels are not in the same order for all images")
                sys.exit()

    # 2.Create column names
    header_row = ["Frame", "Image name", "Cell count"] + ['Stain intensity density, ' + name for name in channels_names]\
                 + ["Total stain intensity, " + name for name in channels_names] + ["Non-nuclear stain intensity, " + name for name in channels_names]

    # 3. Write data
    path = os.path.join(analysis_out_path, analysis_data_folders["analysis"], 'signal_avg_stat.csv')
    with open(path, mode='w', newline='') as stat_file:
        csv_writer = csv.writer(stat_file, delimiter=',')
        csv_writer.writerow(header_row)

        t = 0

        for img_data_t in imgs_data:
            for img_data in img_data_t:
                x = len(channels_names)
                signal_sum_values = np.zeros(len(channels_names))
                cells_total_area = 0
                cell_num = 0

                for i, cell in enumerate(img_data.cells_data):
                    for j in range(0, len(channels_names)):
                        signal_sum_values[j] += cell.signals[j].intensity / cell.area
                    cell_num += 1

                csv_writer.writerow([t, img_data.path, cell_num] + [values / cell_num for values in signal_sum_values] +
                                    [signal for signal in img_data.overall_signal] + [signal for signal in img_data.external_signal])

                csv_writer.writerow([None])

    print("csv avg stat created")

    # Conversion of csv file to xlsx file - removes original csv file

    filepath_in = path
    filepath_out = os.path.join(analysis_out_path, analysis_data_folders["analysis"], 'signal_avg_xlsx.xlsx')
    pd.read_csv(filepath_in, delimiter=",").to_excel(filepath_out, index=False)

    os.remove(filepath_in)


def save_nuc_count_stat(imgs_data_t, save_graph, analysis_out_path):
    """
    Creates a csv sheet and line graph tracking # of nuclei in each image. Mainly useful for timelapses.

    Args:
        imgs_data_t: A list of ImageData objects
        save_graph: A boolean that determines whether a plot tracking nucleus count over time is created
        analysis_out_path: The directory where files will be located after creation

    """

    file_name = os.path.splitext(imgs_data_t[0].path)[0]
    header_row = ["Time point", "Time from experiment start, (min)", "Cell num"]
    path = os.path.join(analysis_out_path, analysis_data_folders["nuclei_count"], file_name +'_time_point_stat.csv')

    nuc_count = []
    time_points = []

    with open(path, mode='w') as stat_file:
        csv_writer = csv.writer(stat_file, delimiter=',')
        csv_writer.writerow(header_row)

        for img_data in imgs_data_t:
            n = img_data.cells_num
            nuc_count.append(n)
            t = img_data.time_point
            time_from_experiment_start = img_data.channels_raw_data[0].time_point
            time_points.append(time_from_experiment_start//60)
            csv_writer.writerow([str(t), str(time_from_experiment_start//60), str(n)])

    if save_graph:
        # plotting the points
        plt.plot(time_points, nuc_count)
        # naming the x axis
        plt.xlabel('time from the beginning of the experiment, min')
        # naming the y axis
        plt.ylabel('cells #')
        # giving a title to my graph
        plt.title('Cell count over time')
        # function to save the plot
        figure_path = os.path.join(analysis_out_path, analysis_data_folders["nuclei_count"], file_name + '_time_point_stat.png')
        plt.savefig(figure_path)

    print(f"Stat for {file_name} is created")


def save_movement_stat(features, analysis_out_path):
    """
    If movement is being tracked in a timelapse, this saves the raw frame, position, and cell id data in an excel sheet

    Args:
        features: A list of pd.DataFrame tables
        analysis_out_path: The directory where the excel sheet will be located after creation

    """

    features.to_excel(os.path.join(analysis_out_path, analysis_data_folders["movement_tracking"], "movement_stat.xlsx"),
                        engine='xlsxwriter') # had to import xlsxwriter for this to work

    print(f"Movement stat is created")

def stitch_mask(input_folder, unet_img_size, num):
    """
    If unet is used for nuc mask, this stitches the parts created by cut_image back together for the full mask

    Args:
        input_folder: The directory where cut mask pieces are located
        unet_img_size: The (length x width) in pixels of the cut unet images. Currently unused? # TODO: Check if needed
        num: An int representing the number of cut pieces that make up the whole mask

    Returns:
        stitched_img: Full 2048x2048 pxl mask for a single image file/frame formed from concatenated 512x512 pieces

    """

    img_col = []
    for i in range(num):
        img_row = []
        for img_path in glob.glob(os.path.join(input_folder, f"*_y-{i}_*.png")):
            nucleus_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_row.append(nucleus_img)
        img_col.append(cv2.hconcat(img_row))
    stitched_img = cv2.vconcat(img_col)
    # img_no_padding = remove_padding(stitched_img) #TODO this function should be created in case img was padded before
    # cv2.imshow("stitched_mask", cv2.resize(stitched_img, (750, 750)))  # keep it for debugging
    # cv2.waitKey()

    return stitched_img


def get_latest_image(dirpath, valid_extensions=('jpg','jpeg','png')):
    """
    Get the last image file (alphabetically) in the given directory

    Args:
        dirpath: Directory where images, including desired last image, are located
        valid_extensions: Valid file types recognized by the function - jpg, jpeg, and png

    Returns:
        The last image file in the given directory, ordered alphabetically

    """

    # get filepaths of all files and dirs in the given dir
    valid_files = [os.path.join(dirpath, filename) for filename in os.listdir(dirpath)]
    # filter out directories, no-extension, and wrong extension files
    valid_files = [f for f in valid_files if '.' in f and \
        f.rsplit('.',1)[-1] in valid_extensions and os.path.isfile(f)]

    if not valid_files:
        raise ValueError("No valid images in %s" % dirpath)

    return max(valid_files, key=os.path.getmtime)


def make_trajectory_fig(trajectory, final_cnt_img, real_t, analysis_out_path):
    """
    For timelapses with movement tracking, creates a figure showing cell movement paths superimposed on the mask of the
    final frame

    Args:
        trajectory: A modified pd.DataFrame table used in timelapse movement tracking
        final_cnt_img: The last mask image (alphabetically) in a given directory, found by get_latest_image
        real_t: The last analyzed frame of the timelapse
        analysis_out_path: The directory where the figure will be located after creation

    """

    fig = plt.figure(figsize=(10, 5))
    tp.plot_traj(trajectory, superimpose=final_cnt_img)
    img_path = os.path.join(analysis_out_path, analysis_data_folders["movement_tracking"],
                            'overall movement - t = ' + str(real_t) + '.png')
    fig.savefig(img_path, bbox_inches='tight', dpi=150)


def plot_movement_trails(features, real_t, analysis_out_path):
    """
    For a timelapse, this calculates and produces movement data for each cell over time, and provides that data to
    make_trajectory_fig to produce a figure.

    Args:
        features: A list of pd.DataFrame tables
        real_t: The last analyzed frame of the timelapse
        analysis_out_path: The directory where the figure will be located after creation

    """

    final_cnt_img = cv2.imread(get_latest_image(os.path.join(analysis_out_path, analysis_data_folders["cnts_verification"])))

    search_range = 100  # Adjustable
    trajectory = tp.link_df(features, search_range, memory=5)  # Memory is Adjustable
    make_trajectory_fig(trajectory, final_cnt_img, real_t,analysis_out_path)

    # Window must be closed to keep the program running TODO: Make figure close automatically?


class Analyzer(object):
    def __init__(self, bioformat_imgs_path, nuc_recognition_mode, nuc_threshold=None, unet_parm=None,
                 nuc_area_min_pixels_num=0, mask_channel_name="DAPI", isWatershed=False, trackMovement=False,
                 trackEachFrame=False, isTimelapse=False, perinuclearArea=False, analysis_out_path=""):
        self.imgs_path = bioformat_imgs_path
        self.nuc_recognition_mode = nuc_recognition_mode
        self.nuc_threshold = nuc_threshold
        self.unet_parm = unet_parm
        self.nuc_area_min_pixels_num = nuc_area_min_pixels_num
        self.mask_channel_name = mask_channel_name
        self.isWatershed = isWatershed
        self.trackMovement = trackMovement
        self.trackEachFrame = trackEachFrame
        self.isTimelapse = isTimelapse
        self.perinuclearArea = perinuclearArea
        self.analysis_out_path = analysis_out_path


    def run_analysis(self):
        """

        Runs the analysis script

        """

        self.analyse_nuc_data()


    def analyse_nuc_data(self):
        """

        Function that carries out the image analysis

        """
        for folder in analysis_data_folders:
            prepare_folder(os.path.join(self.analysis_out_path, analysis_data_folders[folder]))

        imgs_data = []
        features = pd.DataFrame()

        for i, filename in enumerate(os.listdir(self.imgs_path)):
            for folder in temp_folders:
                prepare_folder(temp_folders[folder])
            reader = BioformatReader(self.imgs_path, i, self.mask_channel_name)  # "reader" is a BioformatReader object
            imgs_data_t = []
            # Checks if the provided file is a single image or a timelapse
            if reader.t_num <= 1:
                self.isTimelapse = False
            else:
                self.isTimelapse = True

            # Failsafe in case these weren't changed for non-timelapses in the main()
            if self.isTimelapse is False:
                self.trackMovement = False
                self.trackEachFrame = False

            # real_t can be adjusted to manipulate number of frames in timelapse that are analyzed
            real_t = reader.t_num
            if real_t < 0:
                raise ValueError("'real_t' is less than 0; remember to check real_t when switching between still "
                                 "images and timelapses for analysis")

            for t in range(real_t):
                if self.nuc_recognition_mode == 'unet':
                    nuc_mask = self.find_mask_based_on_unet(reader, t)

                elif self.nuc_recognition_mode == 'thr':
                    nuc_mask = self.find_mask_based_on_thr(reader, t)

                else:
                    print(
                        "The recognition mode is not specified or specified incorrectly. Please use \"unet\" or \"thr\"")
                    sys.exit()

                channels_raw_data = reader.read_all_layers(t)
                img_data = ImageData(filename, channels_raw_data, nuc_mask, self.nuc_area_min_pixels_num, t, self.isWatershed, self.trackMovement, features, self.perinuclearArea)
                features = img_data.features
                img_data.draw_and_save_cnts_for_channels(os.path.join(self.analysis_out_path, analysis_data_folders["cnts_verification"]),
                                                         self.nuc_area_min_pixels_num, self.mask_channel_name, t)
                imgs_data_t.append(img_data)

                # OPTIONAL debugging conditional - for plotting movement trails at every frame in a timelapse
                if self.trackMovement is True and self.trackEachFrame is True:
                    plot_movement_trails(features, t, self.analysis_out_path)

            # Plotting final figure with movement trails
            if self.trackMovement is True:
                plot_movement_trails(features, real_t, self.analysis_out_path)

            # Saving Excel stat files for nuc count and movement
            save_nuc_count_stat(imgs_data_t, save_graph=True, analysis_out_path=self.analysis_out_path)
            if self.trackMovement is True:
                save_movement_stat(features, self.analysis_out_path)

            imgs_data.append(imgs_data_t)
        save_stat(imgs_data, self.isTimelapse, self.analysis_out_path)
        save_avg_stat(imgs_data, self.analysis_out_path)


    def find_mask_based_on_thr(self, reader, t=0):
        """
        Creates nuclear masks through a pixel thresholding mechanism and a Gaussian filter

        Args:
            reader: A BioformatReader object that contains czi image data
            t: The time frame of an image file being analyzed; Mainly relevant for multi-frame timelapses

        Returns:
            nuc_mask: A binary nuclear mask created using the indicated nuclear identification algorithm

        """

        nuc_img_8bit_norm, nuc_file_name = reader.read_nucleus_layers(t=t)

        # produces a Gaussian blurred version of image; kernel is customizable
        gauss_nuc_8bit_norm = cv2.GaussianBlur(nuc_img_8bit_norm, (5, 5), 0)
        _, gauss_nuc_8bit_binary = cv2.threshold(gauss_nuc_8bit_norm, self.nuc_threshold, 255, cv2.THRESH_BINARY)

        nuc_mask = gauss_nuc_8bit_binary

        # all temporary code below - for testing dilation
        # cv2.imshow("original nuc mask - thresholding", cv2.resize(nuc_mask, (750, 750)))
        # kernel = np.ones((5, 5), np.uint8)
        # img_dilation = cv2.dilate(nuc_mask, kernel, iterations=1)
        # cv2.imshow("dilated nuc mask - thresholding", cv2.resize(img_dilation, (750, 750)))
        # nuc_mask = img_dilation
        # cv2.waitKey()

        return nuc_mask

    def find_mask_based_on_unet(self, reader, t=0):
        """
        Finds mask picture based on unet model. Since my GPU can handle only 512*512 images for prediction
        :param reader:
        :return:
        """
        does_cut_img = False
        unet_model_path = self.unet_parm.unet_model_path_63x

        if reader.magnification == "20.0":
            does_cut_img = True
            unet_model_path = self.unet_parm.unet_model_path_20x

        nuc_img_8bit_norm, nuc_file_name = reader.read_nucleus_layers(t=t)
        if does_cut_img:

            pieces_num = cut_image(nuc_img_8bit_norm, nuc_file_name, self.unet_parm.unet_img_size,
                                   temp_folders["cut_8bit_img"])

            run_predict_unet(temp_folders["cut_8bit_img"], temp_folders["cut_mask"],
                             unet_model_path,
                             self.unet_parm.unet_model_scale,
                             self.unet_parm.unet_model_thrh)
            nuc_mask = stitch_mask(temp_folders["cut_mask"], self.unet_parm.unet_img_size, pieces_num)
        else:
            base_img_name = os.path.splitext(os.path.basename(nuc_file_name))[0]
            img_path = os.path.join(temp_folders["cut_8bit_img"], base_img_name + '.png')
            cv2.imwrite(img_path, nuc_img_8bit_norm)

            run_predict_unet(temp_folders["cut_8bit_img"], temp_folders["cut_mask"],
                         unet_model_path,
                         self.unet_parm.unet_model_scale,
                         self.unet_parm.unet_model_thrh)
            img_path = os.path.join(temp_folders["cut_mask"], base_img_name + '.png')
            nuc_mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # DEBUGGING - cv2.imshow("original nuc mask - unet", cv2.resize(nuc_mask, (750, 750)))

        return nuc_mask

    # def _remove_small_particles(self, mask): TODO: Should we remove this? Seems like it's not used
    #
    #
    #     mask = cv2.morphologyEx(mask.astype('uint8'), cv2.MORPH_OPEN, np.ones((5, 5)))
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5)))
    #     clean_mask = np.zeros(mask.shape)
    #     cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    #
    #     for cnt in cnts:
    #         area = cv2.contourArea(cnt)
    #         if area < self.nuc_area_min_pixels_num:
    #             continue
    #         cv2.fillPoly(clean_mask, pts=[cnt], color=(255, 255, 255))
    #     return clean_mask.astype('uint8')
