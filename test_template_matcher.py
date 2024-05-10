from diff_spotter import DiffSpotter
from diff_spotter import Utils
from diff_spotter import TemplateMatcher
import cv2


if __name__ == '__main__':
    images = "images"
    ext = "png"
    fl_nm = "park"
    old_img_path = f"{images}/{fl_nm}_0.{ext}"
    # new_img_path = f"{images}/{fl_nm}_1.{ext}"
    # # https://forum.image.sc/t/align-two-versions-of-the-same-image-that-are-at-different-resolutions-and-one-is-cropped/54737/2
    new_img_path = f"{images}/{fl_nm}_1_small.{ext}"

    tm = TemplateMatcher()
    util = Utils()
    diff = DiffSpotter()

    rvrs_scale = tm.template_matching(old_img_path, new_img_path)
    print(f"\nrvrs_scale {rvrs_scale}")

    old_img = util.load_img(old_img_path)
    new_img = util.load_img(new_img_path)

    new_img = new_img.resize((int(new_img.size[0] * rvrs_scale), int(new_img.size[1] * rvrs_scale)))

    align_new_img = util.align_image(util.PIL_cv2(old_img), util.PIL_cv2(new_img))

    diff_res = diff.spot_diff(old_img, util.cv2_PIL(align_new_img))

    # # cv2로 그리기
    cv2.imshow("result", util.PIL_cv2(diff_res))
    cv2.waitKey(0)
