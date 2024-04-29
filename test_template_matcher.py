from diff_spotter import DiffSpotter
from diff_spotter import Utils
import cv2
import math
import numpy as np
import scipy.ndimage
import imagecodecs
import imreg


def brute_force_scale_invariant_template_matching(
    template,  # grayscale image
    search,  # scaled and cropped grayscale image
    zooms=(1.0, 0.5, 0.25),  # sequence of zoom factors to try
    size=None,  # power-of-two size of square sliding window
    delta=None,  # advance of sliding windows. default: half window size
    min_overlap=0.25,  # minimum overlap of search with window
    max_diff=0.05,  # max average of search - window differences in overlap
    max_angle=0.5,  # no rotation
):
    """Return yoffset, xoffset, and scale of first match of search in template.

    Iterate over scaled versions of the template image in overlapping sliding
    windows and run FFT-based algorithm for translation, rotation and
    scale-invariant image registration until a match of the search image is
    found in the sliding window.

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


def rgb2gray(rgb, scale=None):
    """Return float grayscale image from RGB24 or RGB48 image."""
    scale = np.iinfo(rgb.dtype).max if scale is None else scale
    scale = np.array([[[0.299, 0.587, 0.114]]], np.float32) / scale
    return np.sum(rgb * scale, axis=-1)


def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros((row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray(a, dtype='float32') / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype='uint8')


def template_matching(im1_path, im2_path):
    template = imagecodecs.imread(im1_path)
    search = imagecodecs.imread(im2_path)

    if template.shape[0] * template.shape[1] > search.shape[0] * search.shape[1]:
        tmp = template.copy()
        template = search
        search = tmp

    print(f"{(search.shape[0] * search.shape[1]) / (template.shape[0] * template.shape[1])}")

    multi_n = math.ceil((search.shape[0] * search.shape[1]) / (template.shape[0] * template.shape[1]) * 10) - 10
    print(multi_n)
    template = rgba2rgb(template)
    search = rgba2rgb(search)

    zooms = [1 + zoom / 10 for zoom in range(multi_n + 1)]
    print(f"init zooms {zooms}")
    trial_n = 0
    while True:
        if trial_n == 100: break
        try:
            yoffset, xoffset, scale = brute_force_scale_invariant_template_matching(
                rgb2gray(template), rgb2gray(search), zooms=zooms
            )
            return 1 / scale
        except Exception as err:
            print(f"[ERROR] {err}")
            zooms = [i*1.01 if i != zooms[-1] else i*0.999 for i in zooms]
            if zooms[-2] > zooms[-1]:  # # TODO
                print("Reduce ratio")
                break
            print(zooms)
            trial_n += 1
            pass


if __name__ == '__main__':
    images = "images"
    ext = "png"
    fl_nm = "park"
    old_img_path = f"{images}/{fl_nm}_0.{ext}"
    # new_img_path = f"{images}/{fl_nm}_1.{ext}"
    # # https://forum.image.sc/t/align-two-versions-of-the-same-image-that-are-at-different-resolutions-and-one-is-cropped/54737/2
    new_img_path = f"{images}/{fl_nm}_1_small.{ext}"

    rvrs_scale = template_matching(old_img_path, new_img_path)
    print(f"\nrvrs_scale {rvrs_scale}")

    util = Utils()
    diff = DiffSpotter()

    old_img = util.load_img(old_img_path)
    new_img = util.load_img(new_img_path)

    new_img = new_img.resize((int(new_img.size[0] * rvrs_scale), int(new_img.size[1] * rvrs_scale)))

    align_new_img = util.align_image(util.PIL_cv2(old_img), util.PIL_cv2(new_img))

    diff_res = diff.spot_diff(old_img, util.cv2_PIL(align_new_img))

    # # cv2로 그리기
    cv2.imshow("result", util.PIL_cv2(diff_res))
    cv2.waitKey(0)
