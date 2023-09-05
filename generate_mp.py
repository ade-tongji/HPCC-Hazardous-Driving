import os
import cv2
from tqdm import tqdm
import pandas as pd
import numpy as np

label_csv_path = "/workspace/code/risk_label_headway_close_res_use.csv" #标签文件
video_path = "/workspace/data/res" #原始视频路径
save_path = "/workspace/data/motion_profiles/v14" #保存路径

# motion profile三个区域的分界线，用百分位数[0, 1]，可参考motion_profile_generation.png
TOP_LINE = 0.55
MID_LINE = 0.75
BOTTOM_LINE = 0.85

df = pd.read_csv(label_csv_path)
#print(df)

video_path = "/workspace/data/res"

os.makedirs(f"{save_path}/far")
os.makedirs(f"{save_path}/mid")
os.makedirs(f"{save_path}/near")

for video_name in tqdm(df["saveName"]):
    video_dir = os.path.join(video_path, video_name)
    v_name = os.listdir(video_dir)[0]
    cap = cv2.VideoCapture(os.path.join(video_path, video_name, v_name))
    
    mp_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    mp_h = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    frame_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # 计算生成motion profile三个区域的分界线
    top = round(frame_h * TOP_LINE)
    mid = round(frame_h * MID_LINE)
    bottom = round(frame_h * BOTTOM_LINE)

    # 新建一个宽度为视频宽，高度为视频帧数（时间*FPS）的全0图
    mp_near = np.zeros((int(mp_h), int(mp_w), 3))
    mp_mid = np.zeros((int(mp_h), int(mp_w), 3))
    mp_far = np.zeros((int(mp_h), int(mp_w), 3))

    idx = 0
    while True:

        ret, frame = cap.read()
        #frame = frame.astype(np.float)
        if not ret:
            break
        frame = frame.astype(np.float)
        mp_far[idx, :, 0] = frame[top:mid, :, 0].mean(axis=0)
        mp_far[idx, :, 1] = frame[top:mid, :, 1].mean(axis=0)
        mp_far[idx, :, 2] = frame[top:mid, :, 2].mean(axis=0)
        
        mp_mid[idx, :, 0] = frame[mid:bottom, :, 0].mean(axis=0)
        mp_mid[idx, :, 1] = frame[mid:bottom, :, 1].mean(axis=0)
        mp_mid[idx, :, 2] = frame[mid:bottom, :, 2].mean(axis=0)

        mp_near[idx, :, 0] = frame[bottom:, :, 0].mean(axis=0)
        mp_near[idx, :, 1] = frame[bottom:, :, 1].mean(axis=0)
        mp_near[idx, :, 2] = frame[bottom:, :, 2].mean(axis=0)


        idx += 1
    
    cv2.imwrite(os.path.join(f"{save_path}/far", video_name.split('.')[0] + '.png'), mp_far.astype(np.uint8))
    cv2.imwrite(os.path.join(f"{save_path}/mid", video_name.split('.')[0] + '.png'), mp_mid.astype(np.uint8))
    cv2.imwrite(os.path.join(f"{save_path}/near", video_name.split('.')[0] + '.png'), mp_near.astype(np.uint8))


    cap.release()


