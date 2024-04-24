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
