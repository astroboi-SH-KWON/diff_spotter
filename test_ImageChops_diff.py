from PIL import Image, ImageChops


def loaa_img(img_path):
    return Image.open(img_path)


def spot_diff(img_0, img_1):
    new_x = min(img_0.size[0], img_1.size[0])
    new_y = min(img_0.size[1], img_1.size[1])

    # # RGBA 인 png 파일 등은 RGB 로 변환해 준다.
    img_0 = img_0.crop((0, 0, new_x, new_y)).convert("RGB")
    img_1 = img_1.crop((0, 0, new_x, new_y)).convert("RGB")

    return ImageChops.difference(img_0, img_1)


if __name__ == '__main__':
    images = "images"
    ext = "jpg"
    fl_nm = "sushi"
    old_img_path = f"{images}/{fl_nm}_0.{ext}"
    new_img_path = f"{images}/{fl_nm}_1.{ext}"

    old_img = loaa_img(old_img_path)
    new_img = loaa_img(new_img_path)

    diff_res = spot_diff(old_img, new_img)

    diff_res.save(f"images/{fl_nm}_diff.{ext}")
