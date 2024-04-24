from PIL import ImageChops


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
