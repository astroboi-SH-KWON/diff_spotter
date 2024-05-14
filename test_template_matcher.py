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

    # old_img_path = "images/tmp/img_1.jpg"
    # new_img_path = "images/tmp/img_2.jpg"
    # # https://forum.image.sc/t/align-two-versions-of-the-same-image-that-are-at-different-resolutions-and-one-is-cropped/54737/2
    new_img_path = f"{images}/{fl_nm}_1_small.{ext}"

    tm = TemplateMatcher()
    util = Utils()
    diff = DiffSpotter()

    old_img = util.load_img_by_imagecodecs(old_img_path)
    new_img = util.load_img_by_imagecodecs(new_img_path)

    rvrs_scale = tm.template_matching(old_img, new_img)
    print(f"\nrvrs_scale {rvrs_scale}")

    old_img = util.load_img_by_PIL(old_img_path)
    new_img = util.load_img_by_PIL(new_img_path)

    new_img = new_img.resize((int(new_img.size[0] * rvrs_scale), int(new_img.size[1] * rvrs_scale)))

    align_new_img = util.align_image(util.PIL_cv2(old_img), util.PIL_cv2(new_img))

    diff_res = diff.spot_diff(old_img, util.cv2_PIL(align_new_img))

    diff_res = util.draw_diff_bbox(util.PIL_cv2(diff_res), dark_threshold=45)

    # # cv2로 그리기
    # cv2.imshow("result", util.PIL_cv2(diff_res))
    cv2.imshow("result", diff_res)
    cv2.waitKey(0)
