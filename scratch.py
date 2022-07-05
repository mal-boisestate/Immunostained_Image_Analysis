# Currently unused
import numpy as np
import pandas as pd
import trackpy as tp
import os
from scipy import ndimage
from skimage import morphology, util, filters

# Used
import pims
import cv2.cv2 as cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import skimage

img_example = pims.open("C:\\BioLab\\Immunostained_Image_Analysis\\temp\\full_mask\\22-03-08 overnight DIC DAPI mask 0.png")
# plt.imshow(img_example[0])
plt.imshow(cv2.cvtColor(img_example[0], cv2.COLOR_BGR2RGB))

# Label elements on the picture
black = 0
label_image, number_of_labels = skimage.measure.label(img_example[0], background=black, return_num=True)
print("Found %d features"%(number_of_labels))
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 12))

ax.imshow(img_example[0])

print("ax imshow?")

for region in skimage.measure.regionprops(label_image, intensity_image=img_example[0]):
    # Everywhere, skip small areas
    if region.area < 200:
        continue
    # Only white areas
    if region.mean_intensity < 255:
        continue

    # Draw rectangle which survived to the criterions
    minr, minc, maxr, maxc = region.bbox
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, edgecolor='red', linewidth=1)

    ax.add_patch(rect)

ax.imshow(img_example[0])

