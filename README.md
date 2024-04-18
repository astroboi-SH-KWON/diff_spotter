# diff_spotter
Spot the Difference

    env
        conda create -n astroboi_cv2 python=3.11.4
        pip install opencv-python==4.6.0.66  # cv2는 반드시 pip 설치
        # # python=3.11.4 opencv=4.6.0.66 


    trouble shooting
        1. opencv autocomplete not working on pycharm
            1-0. pip install opencv-python==4.6.0.66  # cv2는 반드시 pip 설치
            1-1. for Mac : copy /Users/{user_name}/anaconda3/envs/{conda_env_name}/lib/python3.11/site-packages/cv2/cv2.abi3.so to /Users/{user_name}/anaconda3/envs/{conda_env_name}/lib/python3.11/site-packages
            1-2. for Windows : copy site-packages/cv2/cv2.pyd to site-packages/

