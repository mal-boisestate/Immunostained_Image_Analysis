class UnetParam(object):
    def __init__(self, unet_model_path_63x, unet_model_path_20x, unet_model_scale, unet_model_thrh, unet_img_size):
        self.unet_model_path_63x = unet_model_path_63x
        self.unet_model_path_20x = unet_model_path_20x
        self.unet_model_scale = unet_model_scale
        self.unet_model_thrh = unet_model_thrh
        self.unet_img_size = unet_img_size


class ImgResolution(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


class ImgChannelTime(object):
    def __init__(self, channel_name, img, time):
        self.name = channel_name
        self.img = img
        self.time_point = time


class Signal(object):
    def __init__(self, channel_name, intensity):
        self.name = channel_name
        self.intensity = intensity


class NucData(object):
    def __init__(self, center, area, perimeter):
        self.area = area
        self.center = center
        self.perimeter = perimeter
        self.signals = None

    def update_signals(self, signals):
        self.signals = signals
