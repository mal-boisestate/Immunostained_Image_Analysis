import numpy as np
import pandas as pd
kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(nuc_mask, kernel, iterations=1)
    img_dilation = cv2.dilate(nuc_mask, kernel, iterations=1)
