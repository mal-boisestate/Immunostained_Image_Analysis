from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience
import pims
import trackpy as tp
import pickle
from objects.ImageData import ImageData
from objects import Contour


#
# with open('test_track_img.pickle', 'rb') as handle:
#     img_data = pickle.load(handle)
#
# frame = img_data.nuc_mask
# centers_coordinates = [Contour.get_cnt_center(center) for center in img_data.cnts]
# # xs = [center[0] for center in centers_coordinates]
# # ys = [center[1] for center in centers_coordinates ]
# d = {'x': [center[0] for center in centers_coordinates], 'y': [center[1] for center in centers_coordinates ]}
# centroids = pd.DataFrame(data=d)
# tp.annotate(centroids, frame)


frames = pims.open(r'D:\BioLab\Current_experiments\cell_tracking\analysis_data_unet_and_watershed\nuclei_area_verification/*.png')
f = tp.locate(frames[0], 11, invert=True)
fs = tp.batch(frames[:5], 11, minmass=20, invert=True)


# tp.annotate(f, frames[0])
a = 1