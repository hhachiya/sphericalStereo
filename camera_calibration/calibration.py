# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import copy
import pdb
from mpl_toolkits.mplot3d import Axes3D

#===================
# function

#-----------------
# 画像上に軸を描画
def draw_origin(img, img_point_origin, imgpts):
    img = cv2.line(img, img_point_origin, tuple(imgpts[0].ravel()), (255,0,0), 5) # x軸
    img = cv2.line(img, img_point_origin, tuple(imgpts[1].ravel()), (0,255,0), 5) # y軸 
    img = cv2.line(img, img_point_origin, tuple(imgpts[2].ravel()), (0,0,255), 5) # z軸
    return img
#-----------------    

#-----------------    
# チェスボードの撮影とキャリブレーション
def calibration(cameraID=1, waitTime=1000, square_size=2.4,corner_size=(7,10),img_num=40,imgPath='images',isVideo=True,prefix='',saveKey=''):
    # チェスボードと画像上のコーナーの座標の記録
    obj_points_all = []
    img_points_all = []

    #-----------------
    # チェスボード（X,Y,Z=0）のコーナーの座標を計算（xy座標は左上原点、z軸は0にあると仮定）
    obj_points = np.zeros( (np.prod(corner_size), 3), np.float32 )
    obj_points_tmp = np.indices(corner_size)
    obj_points[:,:2] = obj_points_tmp.T.reshape(-1, 2)
    obj_points *= square_size
    #-----------------

    #-----------------
    # 画像を撮影し、コーナーの検出
    if isVideo: capture = cv2.VideoCapture(cameraID)

    cnt = 0

    while cnt < img_num:
        # 画像の取得
        if isVideo:
            ret, img = capture.read()
        else:
            img = cv2.imread(f"{imgPath}/{prefix}_captured_{cnt}.png")

        # グレーコードに変換
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # チェスボードのコーナーを検出
        c_flag, corners = cv2.findChessboardCorners(gray, corner_size)

        #-----------------
        # チェスボードのコーナーが検出された場合
        if c_flag == True:            
            # コーナーの高精度化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30 , 0.01)
            corners = cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)

            # コーナーの描画
            img_tmp = copy.deepcopy(img)
            cv2.drawChessboardCorners(img_tmp,corner_size,corners,c_flag)

            if len(saveKey) > 0:
                if cv2.waitKey(200) & 0xFF == ord('s'):                
                    print(f"{cnt+1}/{img_num}")

                    # コーナーの記録
                    img_points_all.append(corners.reshape(-1, 2))
                    obj_points_all.append(obj_points)

                    cv2.imwrite(f"{imgPath}/{prefix}_captured_{cnt}.png",img)
                    cv2.imwrite(f"{imgPath}/{prefix}_detected_corner_{cnt}.png",img_tmp)        
                    cnt += 1
            else:                
                print(f"{cnt+1}/{img_num}")

                # コーナーの記録
                img_points_all.append(corners.reshape(-1, 2))
                obj_points_all.append(obj_points)

                cv2.imwrite(f"{imgPath}/{prefix}_captured_{cnt}.png",img)
                cv2.imwrite(f"{imgPath}/{prefix}_detected_corner_{cnt}.png",img_tmp)        
                cnt += 1

            img = img_tmp
        #-----------------

        # 描画
        cv2.imshow('image', img)
        
        # 200ms待つ
        if cv2.waitKey(waitTime) & 0xFF == ord('q'):
            break
    #-----------------

    # ウィンドウを閉じる
    cv2.destroyAllWindows()

    print("カメラパラメータの計算...")
    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points_all, img_points_all, gray.shape[::-1], None, None)

    print(f"誤差：{rms}")
    print(f"カメラ行列:\n{mtx}")
    print(f"歪みパラメータ：{dist}")

    print("カメラパラメータの保存...")

    # 計算結果を保存
    with open(f"{prefix}_camera_param.pkl",'bw') as fp:
        pickle.dump(mtx,fp) # カメラ行列
        pickle.dump(dist,fp) # 歪みパラメータ
        pickle.dump(rms,fp) # RMS誤差
        pickle.dump(img_points_all,fp)
        pickle.dump(obj_points_all,fp)
#-----------------    
#===================

#===================
# main

if __name__ == '__main__':

    # パラメータの設定
    square_size = 2.41      # 正方形の1辺のサイズ[cm]
    corner_size = (10, 7)  # 交差ポイントの数    
    img_num = 40 # 参照画像の枚数

    # from images
    calibration(square_size=square_size,corner_size=corner_size,img_num=img_num,imgPath='images',isVideo=False,prefix='normal')

    # from video
    #calibration(square_size=square_size,corner_size=corner_size,img_num=img_num,imgPath='images',prefix='normal')

    #-----------------
    # カメラ行列と歪みパラメータの読み込み
    with open('normal_camera_param.pkl','rb') as fp:
        mtx = pickle.load(fp) # カメラ行列
        dist = pickle.load(fp) # 歪みパラメータ
        _ = pickle.load(fp) # RMS誤差
        img_points_all = pickle.load(fp) # 画像のコーナー点
        obj_points_all = pickle.load(fp) # 物体のコーナー点（z=0）    
    #-----------------

    #===================
    # PnP問題を解き、物体（チェッカーボード）の原点を画像座標に射影
    # j: 鏡のindex
    cPij = []
    xPi = []
    for j in range(len(img_points_all)):

        # PnP問題を解き、物体座標系からカメラ画像平面への射影行列（回転と並進）を求める
        ret, rvec, tvec = cv2.solvePnP(obj_points_all[j],img_points_all[j],mtx,dist)

        print(f"mirror {j}:")

        # 回転ベクトル（回転行列のコンパクト表現）
        print(f"rotation:\n{rvec}")

        # 並進ベクトル
        print(f"travel:\n{tvec}")


        # 物体座標系の原点（３軸）の設定
        obj_origin = np.float32([[5.0,0,0], [0,10,0], [0,0,-20]])

        #-----------------
        # 物体座標の原点を画像座標に射影

        # 鏡面画像の読み込み
        img = cv2.imread(f"images/normal_captured_{j}.png")

        obj_origin_img, jac = cv2.projectPoints(obj_origin, rvec, tvec, mtx, dist)
        img = draw_obj_origin(img,tuple(img_points_all[j][0]),obj_origin_img)
        cv2.imwrite(f"images/normal_obj_axis_{j}.png",img)
        cv2.imshow('img',img)
        #-----------------

        #-----------------
        # 物体座標の原点をカメラ座標に射影
        # https://mem-archive.com/2018/02/25/post-201/
        # http://opencv.jp/opencv-2svn/c/camera_calibration_and_3d_reconstruction.html
        # https://mem-archive.com/2018/02/21/post-157/
        # https://stackoverflow.com/questions/44726404/camera-pose-from-solvepnp
        # 物体座標系からカメラ座標系への回転行列
        R = cv2.Rodrigues(rvec)[0]

        # カメラ座標系に物体座標系の原点の位置
        obj_origin_camera = np.dot(R,obj_origin)+tvec
        print(f"obj_origin_camera:\n {obj_origin_camera}")
        #-----------------

        #-----------------
        # 物体座標のコーナー点をカメラ座標に射影
        cPij.append(np.dot(R,obj_points_all[j].T)+tvec)
        #-----------------

        #-----------------
        # 物体座標
        xPi.append(obj_points_all[j].T)
        #-----------------

        cv2.waitKey(0) & 0xFF
    #-----------------

    #-----------------
    # カメラ座標系におけるコーナー点のプロット
    cPi = np.array(cPij)    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(cPij[0][0],cPij[0][1],cPij[0][2],color="red")
    ax.scatter(cPij[1][0],cPij[1][1],cPij[1][2],color="blue")
    ax.scatter(cPij[2][0],cPij[2][1],cPij[2][2],color="green")
    ax.scatter(cPij[3][0],cPij[3][1],cPij[3][2],color="cyan")
    ax.scatter(cPij[4][0],cPij[4][1],cPij[4][2],color="magenta")
    ax.scatter(0,0,0,color="black",s=50)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend(['j=0','j=1','j=2','j=3','j=4'])

    plt.show()
    #-----------------       

    pdb.set_trace()
#===================
