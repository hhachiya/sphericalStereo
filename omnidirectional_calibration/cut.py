import numpy as np
import cv2
from matplotlib import pyplot as plt
import pdb
from matplotlib import pyplot as plt
import pathlib
import pickle
import copy

class sphericalStereo:
    resultPath = "result"
    
    #-----------------
    # コンストラクタ
    def __init__(self):
        # 画像の読み込み
        cnt =  0
        start_ind = 63

        pL = pathlib.Path("stereo29/image")

        
        for imgDir in pL.glob("**/*"):
            img = cv2.imread(f"{imgDir}")

            # 前後の画像に分割
        
            width = int(img.shape[1]/2)
            imgFront = cv2.rotate(img[:,width:], cv2.ROTATE_90_COUNTERCLOCKWISE)
            imgBack = cv2.rotate(img[:,:width], cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(f"stereo29/Back/cut{start_ind:02d}.png",imgBack)
            cv2.imwrite(f"stereo29/Front/cut{start_ind:02d}.png",imgFront)
            cnt2 = cnt2 + 1
        

# Main
sphericalStereo()