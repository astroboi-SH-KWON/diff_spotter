import cv2
from PIL import Image, ImageChops
import numpy as np
import math
import scipy.ndimage
import imagecodecs
import imreg


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
        # # Convert RGBA file like png to RGB
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
        :return: type openCV2
        """
        numpy_image = np.array(PIL_img)
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        return opencv_image

    def cv2_PIL(self, cv2_img):
        """
        Convert openCV2 to PIL.Image
        :param cv2_img: type openCV2
        :return: type PIL.Image
        """
        color_coverted = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_coverted)
        return pil_image

    def align_image(self, im1, im2):
        """
        Align images (HOMOGRAPHY > ECC)
        https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
        :param im1: type openCV2
        :param im2: type openCV2
        :return: type openCV2
        """
        # Convert images to grayscale
        im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        # Find size of image1
        sz = im1.shape

        # Define the motion model
        warp_mode = cv2.MOTION_TRANSLATION

        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Specify the number of iterations.
        number_of_iterations = 5000

        # Specify the threshold of the increment
        # in the correlation coefficient between two iterations
        termination_eps = 1e-10

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)

        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use warpPerspective for Homography
            aligned_im2 = cv2.warpPerspective(im2, warp_matrix, (sz[1], sz[0]),
                                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            # Use warpAffine for Translation, Euclidean and Affine
            aligned_im2 = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]),
                                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        return aligned_im2

    def rgb2gray(self, rgb, scale=None):
        """
        Return float grayscale image from RGB24 or RGB48 image.
        :param rgb: rgb image
        :param scale:
        :return: gray_scale image
        """
        scale = np.iinfo(rgb.dtype).max if scale is None else scale
        scale = np.array([[[0.299, 0.587, 0.114]]], np.float32) / scale
        return np.sum(rgb * scale, axis=-1)

    def rgba2rgb(self, rgba, background=(255, 255, 255)):
        """
        Convert RGBA file like png to RGB
        :param rgba: rgba like png file
        :param background:
        :return: rgb image
        """
        row, col, ch = rgba.shape

        if ch == 3:
            return rgba

        assert ch == 4, 'RGBA image has 4 channels.'

        rgb = np.zeros((row, col, 3), dtype='float32')
        r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

        a = np.asarray(a, dtype='float32') / 255.0

        R, G, B = background

        rgb[:, :, 0] = r * a + (1.0 - a) * R
        rgb[:, :, 1] = g * a + (1.0 - a) * G
        rgb[:, :, 2] = b * a + (1.0 - a) * B

        return np.asarray(rgb, dtype='uint8')

    def draw_diff_bbox(self, diff_img, dark_threshold=25):
        """
        Draw bounding boxes on the spots of differences.
        :param diff_img: type openCV2
        :param dark_threshold: grayscale values range from 0 to 255, with 0 being the darkest.
                            Since values that are too dark (25 or less) are not useful
                            , apply a threshold to consider only values greater than that.
        :return: type openCV2
        """
        gray = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)
        # # remove line noises
        gray = (gray > dark_threshold) * gray

        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        COLOR = (0, 200, 0)

        for cnt in contours:
            if cv2.contourArea(cnt) > 100:  # 사각형 크기가 100 보다 큰 경우에만 그리기
                x, y, width, height = cv2.boundingRect(cnt)
                cv2.rectangle(diff_img, (x, y), (x + width, y + height), COLOR, 2)

        return diff_img


class TemplateMatcher:
    def __init__(self):
        pass

    def brute_force_scale_invariant_template_matching(self,
            template,
            search,
            zooms=(1.0, 0.5, 0.25),
            size=None,
            delta=None,
            min_overlap=0.25,
            max_diff=0.05,
            max_angle=0.5,
    ):
        """
        Iterate over scaled versions of the template image in overlapping sliding
        windows and run FFT-based algorithm for translation, rotation and
        scale-invariant image registration until a match of the search image is
        found in the sliding window.
        :param template: grayscale image
        :param search: scaled and cropped grayscale image
        :param zooms: sequence of zoom factors to try
        :param size: power-of-two size of square sliding window
        :param delta: advance of sliding windows. default: half window size
        :param min_overlap: minimum overlap of search with window
        :param max_diff: max average of search - window differences in overlap
        :param max_angle: no rotation
        :return: yoffset, xoffset, and scale of first match of search in template
        """
        if size is None:
            size = int(pow(2, int(math.log(min(search.shape), 2))))
        if delta is None:
            delta = size // 2
        search = search[:size, :size]
        for zoom in zooms:
            windows = np.lib.stride_tricks.sliding_window_view(
                scipy.ndimage.zoom(template, zoom), search.shape
            )[::delta, ::delta]
            print(windows.shape)
            for i in range(windows.shape[0]):
                for j in range(windows.shape[1]):
                    print(f'{zoom}>>', end='')
                    window = windows[i, j]
                    im2, scale, angle, (t0, t1) = imreg.similarity(window, search)
                    diff = np.abs(im2 - window)[im2 != 0]
                    if (
                            abs(angle) < max_angle
                            and diff.size / window.size > min_overlap
                            and np.mean(diff) < max_diff
                    ):
                        return (
                            (i * delta - t0) / zoom,
                            (j * delta - t1) / zoom,
                            1 / scale / zoom,
                        )
        raise ValueError('no match of search image found in template')

    def template_matching(self, im1_path, im2_path):
        """
        calculate ratio of template matching for 2 images
        :param im1_path:
        :param im2_path:
        :return: ratio of template matching
        """
        utils = Utils()

        template = imagecodecs.imread(im1_path)
        search = imagecodecs.imread(im2_path)

        if template.shape[0] * template.shape[1] > search.shape[0] * search.shape[1]:
            tmp = template.copy()
            template = search
            search = tmp

        print(f"{(search.shape[0] * search.shape[1]) / (template.shape[0] * template.shape[1])}")

        multi_n = math.ceil((search.shape[0] * search.shape[1]) / (template.shape[0] * template.shape[1]) * 10) - 10
        print(multi_n)
        template = utils.rgba2rgb(template)
        search = utils.rgba2rgb(search)

        zooms = [1 + zoom / 10 for zoom in range(multi_n + 1)]
        print(f"init zooms {zooms}")
        trial_n = 0
        while True:
            if trial_n == 100: break
            try:
                yoffset, xoffset, scale = self.brute_force_scale_invariant_template_matching(
                    utils.rgb2gray(template), utils.rgb2gray(search), zooms=zooms
                )
                return 1 / scale
            except Exception as err:
                print(f"[ERROR] {err}")
                zooms = [i * 1.01 if i != zooms[-1] else i * 0.999 for i in zooms]
                if zooms[-2] > zooms[-1]:
                    print("Reduce ratio")
                    break
                print(zooms)
                trial_n += 1
                pass
