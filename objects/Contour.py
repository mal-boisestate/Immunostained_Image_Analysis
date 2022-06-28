import cv2 as cv2
import numpy as np

import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def get_mask_cnts(mask):

    # kernel = np.ones((50, 50), np.uint8)
    #
    # # morphology - opening
    # # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #
    # # morphology - dilation
    # mask = cv2.dilate(mask, kernel, iterations = 1)
    #
    # cv2.imshow("post-morphology mask", cv2.resize(mask, (750, 750)))
    # cv2.waitKey()
    #
    # # Now we want to separate the two objects in image
    # # Generate the markers as local maxima of the distance to the background
    # distance = ndi.distance_transform_edt(mask)
    # coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=mask)
    # temp_mask = np.zeros(distance.shape, dtype=bool)
    # temp_mask[tuple(coords.T)] = True
    # markers, _ = ndi.label(temp_mask)
    # labels = watershed(-distance, markers, mask=mask)
    #
    # fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
    # ax = axes.ravel()
    #
    # ax[0].imshow(mask, cmap=plt.cm.gray)
    # ax[0].set_title('Overlapping objects')
    # ax[1].imshow(-distance, cmap=plt.cm.gray)
    # ax[1].set_title('Distances')
    # ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
    # ax[2].set_title('Separated objects')
    #
    # for a in ax:
    #     a.set_axis_off()
    #
    # fig.tight_layout()
    # plt.show()

    # # kernel creation
    # kernel = np.ones((3, 3), np.uint8)
    #
    # # sure background area
    # sure_bg = cv2.dilate(mask, kernel, iterations=3)
    #
    # cv2.imshow("sure background", cv2.resize(sure_bg, (750, 750)))
    # cv2.waitKey()
    #
    # # Finding sure foreground area
    # dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    #
    # cv2.imshow("sure foreground", cv2.resize(sure_fg, (750, 750)))
    # cv2.waitKey()
    #
    # # Finding unknown region
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv2.subtract(sure_bg, sure_fg)
    #
    # cv2.imshow("unknown region", cv2.resize(unknown, (750, 750)))
    # cv2.waitKey()
    #
    # # Marker labelling
    # ret, markers = cv2.connectedComponents(sure_fg)
    # # Add one to all labels so that sure background is not 0, but 1
    # markers = markers + 1
    # # Now, mark the region of unknown with zero
    # markers[unknown == 255] = 0
    #
    # markers = cv2.watershed(mask, markers)
    # mask[markers == -1] = [255, 0, 0]
    #
    # cv2.imshow("divided mask", cv2.resize(mask, (750, 750)))
    # cv2.waitKey()

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    return cnts


def get_img_cnts(img, theshold):
    """
    Finds and process (remove noise, smooth edges) all contours on the specified image
    ---
        Parameters:
        threshold (int): threshold of pixel intensity starting from which pixels will be labeled as 1 (white)
        img (image): image
    ---
        Returns:
        cnts (vector<std::vector<cv::Point>>): List of detected contours.
                            Each contour is stored as a vector of points.
    """
    _, img_thresh = cv2.threshold(img, theshold, 255, cv2.THRESH_BINARY)

    # Apply pair of morphological "opening" and "closing" to remove noise and to smoothing edges
    img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, np.ones((5, 5)))
    img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, np.ones((5, 5)))

    cnts = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    if len(cnts) == 0:
        return None

    hulls = [cv2.convexHull(cnt, False) for cnt in cnts]
    mask = np.zeros_like(img_thresh)
    cv2.drawContours(mask, hulls, -1, 255, -1)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    return cnts


def draw_cnt(cntr, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [cntr], -1, color=1, thickness=-1)
    return mask


def get_cnt_center(cont):
    if len(cont) <= 2:
        center = (cont[0, 0, 0], cont[0, 0, 1])
    else:
        M = cv2.moments(cont)
        if M["m00"] == 0:
            center_x = int(np.mean([cont[i, 0, 0] for i in range(len(cont))]))
            center_y = int(np.mean([cont[i, 0, 1] for i in range(len(cont))]))
        else:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])

        center = (center_x, center_y)

    return center
