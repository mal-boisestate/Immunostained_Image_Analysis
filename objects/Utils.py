import glob
import os
import numpy as np
import cv2.cv2 as cv2
import math

from objects import Contour

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
        
        
        
def find_optimal_theshold(img, percentile):
    """
    Find what is the minimal intensity of x% pixels that are not null 
    :param img: 
    :return: 
    """
    
    not_zero_pixels = [pixel for pixel in img.flatten() if pixel > 0]
    index = int(percentile / 100 * len(not_zero_pixels))
    opt_threshold = np.sort(not_zero_pixels)[-index]
    return opt_threshold


def remove_small_particles(mask, AREA_TH, theshold = 1):
    mask_8bit = (mask / (math.pow(2, 8))).astype('uint8')
    mask_8bit = cv2.morphologyEx(mask_8bit, cv2.MORPH_OPEN, np.ones((5, 5)))
    mask_8bit = cv2.morphologyEx(mask_8bit, cv2.MORPH_CLOSE, np.ones((5, 5)))
    clean_mask = np.zeros(mask.shape)
    cnts = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < AREA_TH:  # if it is noise not a nuc
            continue
        cv2.fillPoly(clean_mask, pts=[cnt], color=(255, 255, 255))
    return clean_mask.astype('uint8')


def cut_image(img, img_name, unet_img_size, output_folder):
    base_img_name = os.path.splitext(os.path.basename(img_name))[0]
    padded_img_size = (math.ceil(img.shape[0]/ unet_img_size[0]) * unet_img_size[0],
                       math.ceil(img.shape[1] / unet_img_size[1]) * unet_img_size[1])

    padded_img = make_padding(img, padded_img_size)
    y_start = 0
    y_end = unet_img_size[0]
    y_order = 0
    while (padded_img_size[0] - y_end) >= 0:
        x_start = 0
        x_end = unet_img_size[1]
        x_order = 0
        while (padded_img_size[1] - x_end) >= 0:
            current_img = padded_img[y_start:y_end, x_start:x_end]
            img_path = os.path.join(output_folder,
                                    base_img_name + "_y-" + str(y_order) + '_x-' + str(x_order) + '.png')
            cv2.imwrite(img_path, current_img)
            x_start = x_end
            x_end = x_end + unet_img_size[1]
            x_order = x_order + 1
        y_start = y_end
        y_end = y_end + unet_img_size[0]
        y_order = y_order + 1
    return math.ceil(img.shape[0]/ unet_img_size[0])

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


def make_padding(img, final_img_size):
    h, w = img.shape[:2]
    h_out, w_out = final_img_size

    top = (h_out - h) // 2
    bottom = h_out - h - top
    left = (w_out - w) // 2
    right = w_out - w - left

    padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded_img

def cut_nucleus(img, cnt):
    h, w = img.shape

    center = Contour.get_cnt_center(cnt)

    x1, x2 = center[0] - 256, center[0] + 256
    y1, y2 = center[1] - 256, center[1] + 256

    if x1 < 0:
        x1, x2 = 0, 512
    if x2 >= w:
        x1, x2 = w - 1 - 512, w - 1

    if y1 < 0:
        y1, y2 = 0, 512
    if y2 >= h:
        y1, y2 = h - 1 - 512, h - 1

    nucleus_img = img[y1: y2, x1: x2]

    return nucleus_img


def normalization(img, norm_th):
    img[np.where(img > norm_th)] = norm_th
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return img



