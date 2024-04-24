from diff_spotter import DiffSpotter
from diff_spotter import Utils
import cv2


if __name__ == '__main__':
    images = "images"
    ext = "png"
    fl_nm = "park"
    old_img_path = f"{images}/{fl_nm}_0.{ext}"
    new_img_path = f"{images}/{fl_nm}_1.{ext}"

    util = Utils()
    diff = DiffSpotter()

    old_img = util.load_img(old_img_path)
    new_img = util.load_img(new_img_path)

    align_new_img = util.align_image(util.PIL_cv2(old_img), util.PIL_cv2(new_img))

    diff_res = diff.spot_diff(old_img, util.cv2_PIL(align_new_img))

    # # cv2로 그리기
    cv2.imshow("result", util.PIL_cv2(diff_res))
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
