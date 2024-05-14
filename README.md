# diff_spotter
Spot the Difference


    env for web
        conda create -n diff_spotter_web --clone contrib_cv2 
        conda activate diff_spotter_web
        conda install -c anaconda flask==2.2.2

    env
        conda create -n contrib_cv2 python=3.11.4
        pip install opencv-contrib-python==4.6.0.66  # cv2는 반드시 pip 설치, ximgproc 사용위해 opencv-contrib-python 설치
        # # python=3.11.4 opencv=4.6.0.66 
        conda install conda-forge::matplotlib  # v3.8.4
        pip install imreg  # v2024.1.2, similarity for template_matcher


    trouble shooting
        1. opencv autocomplete not working on pycharm
            1-0. pip install opencv-contrib-python==4.6.0.66  # cv2는 반드시 pip 설치
            1-1. for Mac : copy /Users/{user_name}/anaconda3/envs/{conda_env_name}/lib/python3.11/site-packages/cv2/cv2.abi3.so to /Users/{user_name}/anaconda3/envs/{conda_env_name}/lib/python3.11/site-packages
            1-2. for Windows : copy site-packages/cv2/cv2.pyd to site-packages/

