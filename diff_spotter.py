import cv2
from PIL import Image, ImageChops
import numpy as np


class DiffSpotter:
    def __init__(self):
        pass

    def spot_diff(self, img_0, img_1):
        """
        Find spots of differences
        :param img_0: type PIL.Image
        :param img_1: type PIL.Image
        :return: type PIL.Image
        """
        # # RGBA 인 png 파일 등은 RGB 로 변환해 준다.
        return ImageChops.difference(img_0.convert("RGB"), img_1.convert("RGB"))


class Utils:
    def __init__(self):
        pass

    def load_img(self, img_path):
        """
        Load image by PIL.Image
        :param img_path: String
        :return: type PIL.Image
        """
        return Image.open(img_path)

    def PIL_cv2(self, PIL_img):
        """
        Convert PIL.Image to openCV2
        :param PIL_img: type PIL.Image
        :return: openCV2
        """
        numpy_image = np.array(PIL_img)
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        return opencv_image

    def cv2_PIL(self, cv2_img):
        """
        Convert openCV2 to PIL.Image
        :param cv2_img: openCV2
        :return: type PIL.Image
        """
        color_coverted = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_coverted)
        return pil_image
