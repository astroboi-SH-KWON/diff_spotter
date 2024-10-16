from flask import Flask, request, render_template, abort, make_response, Response
from diff_spotter import *
import time
import json
import base64
import uuid
import os
import cv2
import logging


log = logging.getLogger(__name__)


app = Flask('diff_spotter')
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/", methods=['GET', 'POST'])
def diff_spotter_api():
    st_time = time.perf_counter()
    utils = Utils()
    data = request.data.decode('utf-8')
    data = json.loads(data)
    befor_data = data['before_img']
    after_data = data['after_img']
    reduction_ratio = data['reduction_ratio']
    min_size = data['min_size']

    if utils.is_hexadecimal(befor_data) and utils.is_hexadecimal(after_data):
        print("is_hexadecimal >>>>>>>>>>>>>>>>>>>>>>>>")
        befor_bytes = bytes.fromhex(befor_data)
        after_bytes = bytes.fromhex(after_data)
        befor_img = utils.bts_to_img(befor_bytes)
        after_img = utils.bts_to_img(after_bytes)
    elif utils.is_base64(befor_data) and utils.is_base64(after_data):
        print("is_Base64 >>>>>>>>>>>>>>>>>>>>>>>>")
        befor_bytes = base64.b64decode(befor_data)
        after_bytes = base64.b64decode(after_data)
        befor_img = utils.bts_to_img(befor_bytes)
        after_img = utils.bts_to_img(after_bytes)
    else:
        print("[ERROR-diff_spotter_api] Choose decoding options")
        raise Exception

    try:
        now = time.time()
        fl_nm = str(uuid.uuid4())
        os.makedirs("./images/tmp", exist_ok=True)
        utils.remove_files_by_days("./images/tmp", now)
        cv2.imwrite(f"./images/tmp/befor_{fl_nm}.jpg", befor_img)
        cv2.imwrite(f"./images/tmp/after_{fl_nm}.jpg", after_img)

        loaded_befor_img = utils.load_img_by_cv2(f"./images/tmp/befor_{fl_nm}.jpg")
        loaded_after_img = utils.load_img_by_cv2(f"./images/tmp/after_{fl_nm}.jpg")
    except Exception as err:
        log.error(err)
        print(err)
        return abort(make_response(str(err), 500))
    print(f"load_img_by_cv2 ::: {time.perf_counter() - st_time} sec")

    diff_res = get_diff_spotter(loaded_befor_img, loaded_after_img, min_size)

    concat_res = utils.concat_images([loaded_befor_img, diff_res, loaded_after_img])

    _, buffer = cv2.imencode('.jpg', concat_res)
    response = make_response(base64.b64encode(buffer))
    return response


def get_diff_spotter(old_img, new_img, min_size):
    util = Utils()
    diff = DiffSpotter()

    old_img = util.cv2_PIL(old_img)
    new_img = util.cv2_PIL(new_img)

    align_new_img = util.align_image(util.PIL_cv2(old_img), util.PIL_cv2(new_img))

    diff_res = diff.spot_diff(old_img, util.cv2_PIL(align_new_img))

    diff_res = util.draw_diff_bbox(util.PIL_cv2(diff_res), min_size=min_size, dark_threshold=35)

    return diff_res


if __name__ == '__main__':
    port = 8028
    # app.debug = True
    # app.run(debug=True, host='127.0.0.1', port=port)
    app.run(debug=True, host='0.0.0.0', port=port)

"""
sudo lsof -i :8028  
kill -9 {PID}
"""