# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import copy
import sys
import pdb
import calibration
from mpl_toolkits.mplot3d import Axes3D

#===================
# functions
#-----------------
# 3枚の鏡像画像を用いて、カメラと元チェスボードとのキャリブレーション
# 論文：A New Mirror-based Extrinsic Camera CalibrationUsing an Orthogonality Constraint
# Kousuke Takahashi, et al., IPSJ SIG Technical Report, 2012
# http://vision.kuee.kyoto-u.ac.jp/__STATIC__/japanese/happyou/pdf/Takahashi_2012_CVPR.pdf 
def calibrateCameraMirror(cPij,xPi,cInds=[0,9,34,60,69]):
    nj = []
    A = []
    b = []
    ncorners = len(cInds)

    # j: 鏡のindex
    for j in range(len(cPij)):
        Ms = []
        for jj in range(len(cPij)):
            if j==jj:
                continue
            
            # Eq. 7 and 8
            # ２つの鏡像のコーナー点の偏差と偏差平方
            Q = (cPij[j]-cPij[jj]).T            
            Ms.append(np.dot(Q.T,Q))

        # Eq. 8
        # 偏差平方を説明する軸を求める。最小固有値の固有ベクトルが、最も偏差が少ない方向を表している
        # その最も偏差が少ない方向が、２つの鏡像チェッカーの平面が交差する直線（ベクトルmjj'）と考える
        ms = []
        for M in Ms:
            e_value , e_vector = np.linalg.eig(M)
            print(f"minimum eigen value:{np.min(e_value)}")
            ms.append(e_vector[:,np.argmin(e_value)])

        # Eq. 9
        # ２つの平面の直線（mjj'とmjj'')と直交するベクトルが鏡面の法線ベクトルと考える
        mm_cross = np.cross(ms[0],ms[1])[np.newaxis].T
        if mm_cross[2] > 0: mm_cross *= -1
        print(f"cross:\n{mm_cross}")

        # 鏡面の法線ベクトルのz軸はカメラ方向を向いていることを前提に、符号を調整する
        #if mm_cross[2] > 0: mm_cross *= -1
        nj.append(mm_cross/np.linalg.norm(mm_cross))

        # Eq. 14 (computing b)
        b_tmp= -2*np.dot(nj[j].T,cPij[j][:,cInds])*nj[j] + cPij[j][:,cInds]
        b.append(np.reshape(b_tmp.T,[1,-1]))

        # Eq. 14 (computing A)
        if j == 0:
            Aterm1 = np.tile(np.concatenate([np.eye(3),2*nj[j],np.zeros([3,2])],axis=1),[ncorners,1])
        elif j == 1:
            Aterm1 = np.tile(np.concatenate([np.eye(3),np.zeros([3,1]),2*nj[j],np.zeros([3,1])],axis=1),[ncorners,1])
        elif j == 2:
            Aterm1 = np.tile(np.concatenate([np.eye(3),np.zeros([3,2]),2*nj[j]],axis=1),[ncorners,1])            

        Aterm2 = np.repeat(np.repeat(xPi[j][:,cInds].T[:,:2],3,axis=0),3,axis=1)*np.tile(np.eye(3),[ncorners,2])
        A.append(np.concatenate([Aterm1,Aterm2],axis=1))
        
    #-----------------
    # Eq. 10（疑似逆行列を用いてZを求める）
    A = np.concatenate(A,axis=0)
    B = np.concatenate(b,axis=1).T
    Z = np.dot(np.linalg.pinv(A),B)

    # 並進ベクトル
    T = Z[:3]

    # 距離
    D = Z[3:6]

    # 回転行列の計算（r3の計算と正規化）
    R = np.concatenate([Z[6:9],Z[9:12]],axis=1)
    r3 = np.cross(R[:,0],R[:,1])/(np.linalg.norm(R[:,0])*np.linalg.norm(R[:,1]))
    #r2 = np.cross(r3,R[:,0])/(np.linalg.norm(r3)*np.linalg.norm(R[:,0]))
    #r1 = R[:,0]/np.linalg.norm(R[:,0])
    R = np.concatenate([R,r3[np.newaxis].T],axis=1)

    U,S,V=np.linalg.svd(R)
    R = np.dot(U,V.T)
    #-----------------

    return T, R, D, nj, Z
#----------------- 
#===================

#===================
# main

# パラメータの設定
square_size = 2.41      # 正方形の1辺のサイズ[cm]
corner_size = (10, 7)  # 交差ポイントの数
img_num = 5 # 参照画像の枚数

# from images
#calibration.calibration(waitTime=100, square_size=square_size,corner_size=corner_size,img_num=img_num,imgPath='images',isVideo=False, prefix='mirrored',saveKey='s')

# from video
#calibration.calibration(waitTime=100, square_size=square_size,corner_size=corner_size,img_num=img_num,imgPath='images',isVideo=True, prefix='mirrored',saveKey='s')

#-----------------
# カメラ行列と歪みパラメータの読み込み
with open('normal_camera_param.pkl','rb') as fp:
    mtx = pickle.load(fp) # カメラ行列
    dist = pickle.load(fp) # 歪みパラメータ
    _ = pickle.load(fp) # RMS誤差
#-----------------

#-----------------
# 鏡像のコーナー点の読み込み
with open('mirrored_camera_param.pkl','rb') as fp:
    _ = pickle.load(fp) # カメラ行列
    _ = pickle.load(fp) # 歪みパラメータ
    _ = pickle.load(fp) # RMS誤差
    img_points_all = pickle.load(fp) # 画像のコーナー点
    obj_points_all = pickle.load(fp) # 物体のコーナー点（z=0）
#-----------------

#-----------------
# 鏡像のチェスボード（X,Y,Z=0）のコーナーの座標を計算（左右反転しているため、xy座標は右上原点、z軸は0にあると仮定）
obj_points_mirrored = np.zeros( (np.prod(corner_size), 3), np.float32 )
obj_points_mirrored_tmp = np.indices(corner_size)

# x軸を反転
obj_points_mirrored_tmp[0] = obj_points_mirrored_tmp[0][::-1]
obj_points_mirrored[:,:2] = obj_points_mirrored_tmp.T.reshape(-1, 2)
obj_points_mirrored *= square_size
#-----------------

#-----------------
# 座標の原点（３軸）の設定
origin = np.float32([[5.0,0,0], [0,10,0], [0,0,-20]])
#-----------------

#===================
# PnP問題を解き、カメラ座標系における鏡像の座標cPijを求める
# j: 鏡のindex
cPij = []
xPi = []
for j in range(len(img_points_all)):

    # PnP問題を解き、鏡像の物体座標系からカメラ画像平面への射影行列（回転と並進）を求める
    ret, rvec, tvec = cv2.solvePnP(obj_points_all[j],img_points_all[j],mtx,dist)

    print(f"mirror {j}:")

    # 回転ベクトル（回転行列のコンパクト表現）
    print(f"rotation:\n{rvec}")

    # 並進ベクトル
    print(f"travel:\n{tvec}")


    #-----------------
    # 物体座標の原点を画像座標に射影

    # 鏡面画像の読み込み
    img = cv2.imread(f"images/mirrored_captured_{j}.png")

    obj_origin_img, jac = cv2.projectPoints(origin, rvec, tvec, mtx, dist)
    img = calibration.draw_origin(img,tuple(img_points_all[j][0]),obj_origin_img)
    cv2.imwrite(f"images/mirrored_obj_axis_{j}.png",img)
    cv2.imshow('img',img)
    #-----------------

    #-----------------
    # 物体座標のコーナー点をカメラ座標に射影
    # https://mem-archive.com/2018/02/25/post-201/
    # http://opencv.jp/opencv-2svn/c/camera_calibration_and_3d_reconstruction.html
    # https://mem-archive.com/2018/02/21/post-157/
    # https://stackoverflow.com/questions/44726404/camera-pose-from-solvepnp

    # 物体座標系からカメラ座標系への回転行列
    R = cv2.Rodrigues(rvec)[0]
    cPijtmp = np.dot(R,obj_points_mirrored.T)+tvec
    #cPijtmp = np.dot(R,obj_points_all[j].T)+tvec  
    cPij.append(cPijtmp)
    #-----------------

    #-----------------
    # 物体座標
    xPi.append(obj_points_all[j].T)
    #-----------------

    cv2.waitKey(0) & 0xFF
#===================

#-----------------
# 鏡像を用いた、物体とカメラ間の外部パラメータR, Tの推定
cPij = np.array(cPij)
xPi = np.array(xPi)
mirror_inds = [0,1,3] # ３つのチェッカーボード間の境界線が平行の場合、外積が０になる
T, R, D, nj, Z = calibrateCameraMirror(cPij[mirror_inds],xPi[mirror_inds],cInds=np.arange(70))
cPi = np.dot(R,xPi[0])+T
#-----------------

#-----------------
# 鏡面の計算
mirrors= []
for ind, mInd in enumerate(mirror_inds):
    tj = -np.dot(nj[ind].T,cPij[mInd]) - D[ind]
    mirrors.append(cPij[mInd] + nj[ind]*tj)
#-----------------


#-----------------
# カメラ座標系におけるコーナー点のプロット
fig = plt.figure()
ax = Axes3D(fig)

# 鏡像
ax.scatter(cPij[mirror_inds[0]][0],cPij[mirror_inds[0]][1],cPij[mirror_inds[0]][2],color="red")
ax.scatter(cPij[mirror_inds[1]][0],cPij[mirror_inds[1]][1],cPij[mirror_inds[1]][2],color="blue")
ax.scatter(cPij[mirror_inds[2]][0],cPij[mirror_inds[2]][1],cPij[mirror_inds[2]][2],color="green")

# 鏡面
ax.scatter(mirrors[0][0],mirrors[0][1],mirrors[0][2],marker="*",alpha=0.5,color="red")
ax.scatter(mirrors[1][0],mirrors[1][1],mirrors[1][2],marker="*",alpha=0.5,color="blue")
ax.scatter(mirrors[2][0],mirrors[2][1],mirrors[2][2],marker="*",alpha=0.5,color="green")

# 物体
ax.scatter(xPi[0][0],xPi[0][1],xPi[0][2],color="magenta")
ax.scatter(cPi[0],cPi[1],cPi[2],color="cyan")

# 各原点
ax.scatter(cPij[mirror_inds[0]][0][0],cPij[mirror_inds[0]][1][0],cPij[mirror_inds[0]][2][0],color="red",s=50)
ax.scatter(cPij[mirror_inds[1]][0][0],cPij[mirror_inds[1]][1][0],cPij[mirror_inds[1]][2][0],color="blue",s=50)
ax.scatter(cPij[mirror_inds[2]][0][0],cPij[mirror_inds[2]][1][0],cPij[mirror_inds[2]][2][0],color="green",s=50)
ax.scatter(xPi[0][0][0],xPi[0][1][0],xPi[0][2][0],color="magenta",s=50)
ax.scatter(cPi[0][0],cPi[1][0],cPi[2][0],color="cyan",s=50)
ax.scatter(0,0,0,color="black",s=50)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend([f'mirrored {mirror_inds[0]}',f'mirrored {mirror_inds[1]}',f'mirrored {mirror_inds[2]}',f'mirror {mirror_inds[0]}',f'mirror {mirror_inds[1]}',f'mirror {mirror_inds[2]}','object','object in camera space'])

plt.savefig('mirrored_calibrated_object.png')
plt.show()
pdb.set_trace()
#-----------------
#===================
