import cv2
from PIL import Image, ImageChops
import numpy as np
import matplotlib.pyplot as plt


def load_img(img_path):
    return Image.open(img_path)


def PIL_cv2(PIL_img):
    numpy_image = np.array(PIL_img)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return opencv_image


def cv2_PIL(cv2_img):
    color_coverted = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_coverted)
    return pil_image


def spot_diff(img_0, img_1):
    # # RGBA 인 png 파일 등은 RGB 로 변환해 준다.
    return ImageChops.difference(img_0.convert("RGB"), img_1.convert("RGB"))


def align_image(im1, im2):
    # # https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
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
        aligned_im2 = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return aligned_im2


if __name__ == '__main__':
    images = "images"
    ext = "png"
    fl_nm = "park"
    old_img_path = f"{images}/{fl_nm}_0.{ext}"
    new_img_path = f"{images}/{fl_nm}_1.{ext}"

    old_img = load_img(old_img_path)
    new_img = load_img(new_img_path)

    align_new_img = align_image(PIL_cv2(old_img), PIL_cv2(new_img))

    diff_res = spot_diff(old_img, cv2_PIL(align_new_img))

    # # cv2로 그리기
    cv2.imshow("result", PIL_cv2(diff_res))
    cv2.waitKey(0)

    # # # plt 으로 그리기
    # fig, axs = plt.subplots(1, 4, figsize=(10, 5))
    #
    # axs[0].imshow(old_img)
    # axs[0].axis('off')  # x, y축 제거
    #
    # axs[1].imshow(new_img)
    # axs[1].axis('off')  # x, y축 제거
    #
    # axs[2].imshow(align_new_img)
    # axs[2].axis('off')  # x, y축 제거
    #
    # axs[3].imshow(diff_res)
    # axs[3].axis('off')  # x, y축 제거
    #
    # plt.show()
