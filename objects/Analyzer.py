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

def save_stat(imgs_data, isTimelapse, analysis_out_path): # TODO: Make this function work - plan is to create a separate stat file for each timelapse
    """
    Extract and save statistical data for a timelapse
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
                  "Nucleus area, pixels"] + ['Total intensity, ' + name for name in channels_names] + \
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
                    csv_writer.writerow([t, img_data.path, str(i), str(cell.center), str(cell.area)] +
                                        [signal.intensity for signal in cell.signals] +
                                        [signal.intensity/cell.area for signal in cell.signals])
                if isTimelapse is True:
                    t += 1
                csv_writer.writerow([None, None, None, None, None] +
                                 [None for signal in cell.signals])

    print("csv stat created")

    # TODO: xlsx approach

    filepath_in = path
    filepath_out = os.path.join(analysis_out_path, analysis_data_folders["analysis"], 'signal_quant_xlsx.xlsx')
    pd.read_csv(filepath_in, delimiter=",").to_excel(filepath_out, index=False)

    os.remove(filepath_in)

def save_nuc_count_stat(imgs_data_t, save_graph, analysis_out_path):
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


def save_movement_stat(features, analysis_out_path): # IN PROGRESS
    # header_row = ["Time point", "Time from experiment start, (min)", "Cell num"]
    # path = os.path.join(analysis_data_folders["analysis"], file_name +'_movement_stat.csv')

    features.to_excel(os.path.join(analysis_out_path, analysis_data_folders["movement_tracking"], "movement_stat.xlsx"),
                        engine='xlsxwriter') # had to import xlsxwriter for this to work

    print(f"Movement stat is created")

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

def get_latest_image(dirpath, valid_extensions=('jpg','jpeg','png')):
    """
    Get the latest image file in the given directory
    """

    # get filepaths of all files and dirs in the given dir
    valid_files = [os.path.join(dirpath, filename) for filename in os.listdir(dirpath)]
    # filter out directories, no-extension, and wrong extension files
    valid_files = [f for f in valid_files if '.' in f and \
        f.rsplit('.',1)[-1] in valid_extensions and os.path.isfile(f)]

    if not valid_files:
        raise ValueError("No valid images in %s" % dirpath)

    return max(valid_files, key=os.path.getmtime)

# def centralize_trajectories(trajectory):
#
#     particle_num = 0
#
#     for particle in trajectory:
#         for particles in particle:
#             if particles == particle_num:
#


def make_trajectory_fig(trajectory, final_cnt_img, real_t, analysis_out_path):

    fig = plt.figure(figsize=(10, 5))
    tp.plot_traj(trajectory, superimpose=final_cnt_img)
    img_path = os.path.join(analysis_out_path, analysis_data_folders["movement_tracking"],
                            'overall movement - t = ' + str(real_t) + '.png')
    fig.savefig(img_path, bbox_inches='tight', dpi=150)

def plot_movement_trails(features, real_t, analysis_out_path):
    final_cnt_img = cv2.imread(get_latest_image(os.path.join(analysis_out_path, analysis_data_folders["cnts_verification"])))

    search_range = 100  # Adjustable
    trajectory = tp.link_df(features, search_range, memory=5)  # Memory is Adjustable
    make_trajectory_fig(trajectory, final_cnt_img, real_t,analysis_out_path)

    # Window must be closed to keep the program running TODO: Make figure close automatically?

    # IDEA: Save the DataFrames and create every plot at the end of the analysis?
    # Would still have the same problem, concentrated at the end... might still be a decent workaround

    # 7/22 PLAN: - nvm ask Nina
    # 1) Take the trajectories data made from tp.link_df - where particles are already labeled
    # 2) Find displacement to take all coordinates for each particle origin to center (1024, 1024) - specific to each particle
    # 3) Once every coordinate is updated, superimpose this on the plot of choice

class Analyzer(object):
    def __init__(self, bioformat_imgs_path, nuc_recognition_mode, nuc_threshold=None, unet_parm=None,
                 nuc_area_min_pixels_num=0, mask_channel_name="DAPI", isWatershed=False, trackMovement=False,
                 trackEachFrame=False, isTimelapse=False, analysis_out_path=""):
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
        self.analysis_out_path = analysis_out_path


    def run_analysis(self):
        self.analyse_nuc_data()


    def analyse_nuc_data(self):

        for folder in analysis_data_folders:
            prepare_folder(os.path.join(self.analysis_out_path, analysis_data_folders[folder]))  # prepares analysis data folder

        imgs_data = []
        features = pd.DataFrame()  # Contains identified objects and their locations at each frame

        for i, filename in enumerate(os.listdir(self.imgs_path)):
            for folder in temp_folders:
                prepare_folder(temp_folders[folder])
            reader = BioformatReader(self.imgs_path, i, self.mask_channel_name)  # "reader" is a BioformatReader object,
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
                # NEW movement tracking goes through ImageData as follows
                img_data = ImageData(filename, channels_raw_data, nuc_mask, self.nuc_area_min_pixels_num, t, self.isWatershed, self.trackMovement, features)
                features = img_data.features
                img_data.draw_and_save_cnts_for_channels(os.path.join(self.analysis_out_path, analysis_data_folders["cnts_verification"]),
                                                         self.nuc_area_min_pixels_num, self.mask_channel_name, t)
                imgs_data_t.append(img_data)

                # OPTIONAL - for plotting movement trails at every frame in a timelapse
                if self.trackMovement is True and self.trackEachFrame is True:
                    plot_movement_trails(features, t, self.analysis_out_path)

            # Plotting final figure with movement trails
            if self.trackMovement is True:
                plot_movement_trails(features, real_t, self.analysis_out_path)

            # Saving Excel stat files for nuc count and movement
            save_nuc_count_stat(imgs_data_t, save_graph=True, analysis_out_path=self.analysis_out_path)
            if self.trackMovement is True:
                save_movement_stat(features, self.analysis_out_path)

            imgs_data.append(imgs_data_t) # Here, every imgs_data_t object represents a list of ImageData objects at each time point in a timelapse

        save_stat(imgs_data, self.isTimelapse, self.analysis_out_path) # TODO: Make this function work and replace old stat writing - imgs_data is now a list of imgs_data_t, each of which is a list of ImageData objects


    def find_mask_based_on_thr(self, reader, t=0):
        # Look at self._remove_small_particles function it can be helpful - might not need though?
        # Logic flow: noise reduction (Gaussian filter) -> binary thresholding (0/255 scale) -> returning nuc_mask
        # _remove_small_particles function - is it needed?

        # reads czi image from reader and normalizes to 8bit image - bypassing need for conditional thresholding
        nuc_img_8bit_norm, nuc_file_name = reader.read_nucleus_layers(t=t)

        # produces a Gaussian blurred version of image; kernel is customizable (how to confirm?)
        gauss_nuc_8bit_norm = cv2.GaussianBlur(nuc_img_8bit_norm, (5, 5), 0)

        # modifies each pixel in the blurred image, creating a binary image based on the user-provided threshold
        # for x in range(0, len(gauss_nuc_8bit_norm)):
        #    for y in range(0, len(gauss_nuc_8bit_norm)):
        #        if gauss_nuc_8bit_norm[x][y] < self.nuc_threshold:
        #            gauss_nuc_8bit_norm[x][y] = 0
        #        else:
        #            gauss_nuc_8bit_norm[x][y] = 255

        # built-in binary functionality of numpy returns true/false array, which doesn't seem to work:
        # gauss_nuc_8bit_bin = gauss_nuc_8bit_norm < self.nuc_threshold

        # ALTERNATIVELY, could use built-in thresholding function - requires conversion to greyscale image:

        # gauss_nuc_8bit_grey = cv2.cvtColor(gauss_nuc_8bit_norm, cv2.COLOR_BGR2GRAY)
        _, gauss_nuc_8bit_binary = cv2.threshold(gauss_nuc_8bit_norm, self.nuc_threshold, 255, cv2.THRESH_BINARY)

        # the following lines are for debugging, and MAY BE TURNED OFF to prevent them from popping up
        # cv2.imshow("original img", cv2.resize(nuc_img_8bit_norm, (750, 750)))
        # cv2.imshow("gaussian blurred img", cv2.resize(gauss_nuc_8bit_norm, (750, 750)))  # keep it for debugging
        # cv2.imshow("gaussian binary img", cv2.resize(gauss_nuc_8bit_binary, (750, 750)))  # keep it for debugging
        # cv2.waitKey()

        nuc_mask = gauss_nuc_8bit_binary
        cv2.waitKey()

        return nuc_mask

        # ALTERNATIVELY, could use built-in thresholding function - requires conversion to greyscale image (nope!)

    def find_mask_based_on_unet(self, reader, t=0):
        """
        Finds mask picture based on unet model. Since my GPU can handle only 512*512 images for prediction
        :param reader:
        :return:
        """
        nuc_img_8bit_norm, nuc_file_name = reader.read_nucleus_layers(t=t)
        pieces_num = cut_image(nuc_img_8bit_norm, nuc_file_name, self.unet_parm.unet_img_size,
                               temp_folders["cut_8bit_img"])

        run_predict_unet(temp_folders["cut_8bit_img"], temp_folders["cut_mask"],
                         self.unet_parm.unet_model_path,
                         self.unet_parm.unet_model_scale,
                         self.unet_parm.unet_model_thrh)
        nuc_mask = stitch_mask(temp_folders["cut_mask"], self.unet_parm.unet_img_size, pieces_num)
        # cv2.imshow("original nuc mask - unet", cv2.resize(nuc_mask, (750, 750)))

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