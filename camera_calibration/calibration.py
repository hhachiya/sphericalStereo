# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pdb

# パラメータの設定
square_size = 2.4      # 正方形の1辺のサイズ[cm]
corner_size = (7, 10)  # 交差ポイントの数
img_num = 40 # 参照画像の枚数

# チェスボード（X,Y,Z=0）のコーナーの座標を計算（xy座標は左上原点、z軸は0にあると仮定）
obj_corner = np.zeros( (np.prod(corner_size), 3), np.float32 )
obj_corner[:,:2] = np.indices(corner_size).T.reshape(-1, 2)
obj_corner *= square_size

# チェスボードと画像上のコーナーの座標の記録
obj_corners = []
img_corners = []

# 結果画像の保存先
imgPath = 'images'

# キャプチャー開始
capture = cv2.VideoCapture(0)
cnt = 0

while cnt < img_num:
    # 画像の取得
    ret, img = capture.read()

    # グレーコードに変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # チェスボードのコーナーを検出
    c_flag, corners = cv2.findChessboardCorners(gray, corner_size)

    #-----------------
    # チェスボードのコーナーが検出された場合
    if c_flag == True:
        print(f"{cnt+1}/{img_num}")
        
        # コーナーの高精度化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30 , 0.01)
        #corners = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        corners = cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)

        # コーナーの記録
        img_corners.append(corners.reshape(-1, 2))
        obj_corners.append(obj_corner)

        # コーナーの描画
        cv2.drawChessboardCorners(img,corner_size,corners,c_flag)        
        cv2.imwrite(f"{imgPath}/detected_corner_{cnt}.png",img)        
        cnt += 1
    #-----------------

    # 描画
    cv2.imshow('image', img)
    
    # 200ms待つ
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

print("カメラパラメータの計算...")
rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_corners, img_corners, gray.shape[::-1], None, None)

print(f"誤差：{rms}")
print(f"カメラ行列:\n{mtx}")
print(f"歪みパラメータ：{dist}")

print("カメラパラメータの保存...")

# 計算結果を保存
with open('camera_param.pkl','bw') as fp:
    pickle.dump(mtx,fp) # カメラ行列
    pickle.dump(dist,fp) # 歪みパラメータ
    pickle.dump(rms,fp) # RMS誤差

pdb.set_trace()
