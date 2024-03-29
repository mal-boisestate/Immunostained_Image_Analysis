import cv2 as cv2
import numpy as np

def get_mask_cnts(mask):

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
