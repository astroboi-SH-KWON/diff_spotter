from diff_spotter import DiffSpotter
from diff_spotter import Utils
from diff_spotter import TemplateMatcher
from flask import Flask, render_template, request, make_response, abort
import cv2
import numpy as np
import time
import base64
import os
import logging


log = logging.getLogger(__name__)

app = Flask(__name__)


@app.route("/")
def home():
    log.info("START ::::::::::::::::::::::")
    print("START ::::::::::::::::::::::")
    return render_template('view.html')


@app.route('/uploadajax', methods=['POST'])
def upldfile():
    if request.method == 'POST':
        util = Utils()

        reduction_ratio = request.form.get("reduction_ratio", type=float)
        # read image file string data
        file_1 = request.files['file_1'].read()
        file_2 = request.files['file_2'].read()

        # # convert string data to numpy array
        file_1_bytes = np.frombuffer(file_1, dtype=np.uint8)
        file_2_bytes = np.frombuffer(file_2, dtype=np.uint8)

        decd_img_1 = cv2.imdecode(file_1_bytes, cv2.IMREAD_COLOR)
        decd_img_2 = cv2.imdecode(file_2_bytes, cv2.IMREAD_COLOR)

        now = time.time()
        try:
            os.makedirs("./images/tmp", exist_ok=True)
            util.remove_files_by_days("./images/tmp", now)
            cv2.imwrite(f"./images/tmp/img_1_{now}.jpg", decd_img_1)
            cv2.imwrite(f"./images/tmp/img_2_{now}.jpg", decd_img_2)

            img_1 = util.load_img_by_imagecodecs(f"./images/tmp/img_1_{now}.jpg")
            img_2 = util.load_img_by_imagecodecs(f"./images/tmp/img_2_{now}.jpg")
        except Exception as err:
            log.error(err)
            print(err)
            return abort(make_response(str(err), 500))

        diff_res = get_diff_spotter(img_1, img_2, reduction_ratio)

        concat_res = util.concat_images([decd_img_1, diff_res, decd_img_2])

        _, buffer = cv2.imencode('.jpg', concat_res)
        response = make_response(base64.b64encode(buffer))
        response.headers.set('Content-Type', 'image/gif')
        response.headers.set('Content-Disposition', 'attachment', filename='image.gif')
        return response


def get_diff_spotter(old_img, new_img, reduction_ratio):
    tm = TemplateMatcher()
    util = Utils()
    diff = DiffSpotter()

    rvrs_scale = tm.template_matching(old_img, new_img, reduction_ratio=reduction_ratio)
    log.info(f"\nrvrs_scale {rvrs_scale}")
    print(f"\nrvrs_scale {rvrs_scale}")

    old_img = util.cv2_PIL(old_img)
    new_img = util.cv2_PIL(new_img)
    new_img = new_img.resize((int(new_img.size[0] * rvrs_scale), int(new_img.size[1] * rvrs_scale)))

    align_new_img = util.align_image(util.PIL_cv2(old_img), util.PIL_cv2(new_img))

    diff_res = diff.spot_diff(old_img, util.cv2_PIL(align_new_img))

    diff_res = util.draw_diff_bbox(util.PIL_cv2(diff_res), dark_threshold=35)

    return diff_res


if __name__ == "__main__":
    port = 8028
    # app.debug = True
    # app.run(debug=True, host='127.0.0.1', port=port)
    app.run(debug=True, host='0.0.0.0', port=port)  # http://127.0.0.1:8028/


"""
sudo lsof -i :8028  
kill -9 {PID}
"""
