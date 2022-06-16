import os
from pathlib import Path
import bioformats
import cv2 as cv2
import numpy as np
from objects.Structures import ImgResolution, PairImgChannel


class BioformatReader(object):
    """
    Creates an object that reads confocal microscopy images of two channels (actin and nucleus)
    """

    def __init__(self, path, img_number, mask_channel_name):
        """
            Parameters:
            img_path (string): path to the file to read
            nucleus_channel(int): channel of nucleus images at the provided microscopic image
            actin_channel(int): channel of actin images at the provided microscopic image
        """
        self.image_path, self.series = self.get_img_path_and_series(path, img_number)
        metadata = bioformats.get_omexml_metadata(str(self.image_path)) # Obtaining image metadata
        self.metadata_obj = bioformats.OMEXML(metadata) # Creation of a metadata "object"?
        self.channel_nums = self.metadata_obj.image(self.series).Pixels.get_channel_count()
        self.channels = self.find_channels()
        self.nuc_channel = self.find_channel(mask_channel_name) # Intended to use nuclear channel specifically?
        self.img_resolution = self.get_resolution()
        self.depth = self.metadata_obj.image(self.series).Pixels.PixelType # Where depth (8 bit, etc.) is identified?
                                                                           # What does this actually return?

    def find_channels(self):
        channels = {}
        for i in range(self.channel_nums):
            channels[i] = self.metadata_obj.image().Pixels.Channel(i).get_Name()
        return channels

    def find_channel(self, channel_name):
        channel_num = None
        for i in range(self.channel_nums):
            if self.metadata_obj.image().Pixels.Channel(i).get_Name() == channel_name:
                channel_num = i

        return channel_num

    def get_all_channels_names(self):
        names = []
        for i in range(self.channel_nums):
            names.append(self.metadata_obj.image().Pixels.Channel(i).get_Name())
        return names

    def get_resolution(self):
        scale_x = self.metadata_obj.image(self.series).Pixels.get_PhysicalSizeX()
        scale_y = self.metadata_obj.image(self.series).Pixels.get_PhysicalSizeY()
        img_resolution = ImgResolution(scale_x, scale_y)
        return img_resolution

    def get_img_path_and_series(self, path, cell_number):
        """
        CZI and LIF files, in our case organized differently.
        LIF is a project file that has different images as a Series.
        CZI is a path to the folder that contains separate images.
        This method checks what is the case and finds the path-specific image and Series.
        Args:
            path: str, path to folder or project file

        Returns:
            img_path: path to file
            series: series to analyze
        """
        img_path = None
        if os.path.isfile(path):
            series = cell_number
            img_path = path

        else:
            series = 0
            folder_path = path
            for i, current_path in enumerate(Path(folder_path).rglob('*.czi')):
                if i == cell_number:
                    img_path = current_path
                    break

        return img_path, series

    def read_all_layers(self):
        pairs_img_channel = []
        for channel in self.channels:
            img = bioformats.load_image(str(self.image_path), c=channel, z=0, t=0, series=self.series, index=None,
                                        rescale=False,
                                        wants_max_intensity=False,
                                        channel_names=None)
            pairs_img_channel.append(PairImgChannel(self.channels[channel], img))
        return pairs_img_channel

    def read_nucleus_layers(self, norm=True):
        base_img_name = os.path.splitext(os.path.basename(self.image_path))[0]
        img = bioformats.load_image(str(self.image_path), c=self.nuc_channel, z=0, t=0, series=self.series, index=None,
                                    rescale=False,
                                    wants_max_intensity=False,
                                    channel_names=None)
        if norm:
            threshold = self.find_optimal_theshold(img, percentile=0.01)
            img = self.normalization(img, threshold)

        return img, base_img_name + '_' + self.channels[self.nuc_channel] + '.png'

    def find_optimal_theshold(self, img, percentile):
        """
        Find what is the minimal intensity of x% pixels that are not null
        :param img:
        :return:
        """

        not_zero_pixels = [pixel for pixel in img.flatten() if pixel > 0]
        index = int(percentile / 100 * len(not_zero_pixels))
        opt_threshold = np.sort(not_zero_pixels)[-index]
        return opt_threshold

    def normalization(self, img, norm_th):
        img[np.where(img > norm_th)] = norm_th
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        return img

    def close(self):
        bioformats.clear_image_reader_cache()
