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
# 画像からコーナーの検出
def detectCornerFromImages(cameraID=1, waitTime=1000, corner_size=(7,10), img_num=40, img_w=1280, img_h=720, imgPath='images', mask_imgs=[], isVideo=True, load_prefix='',  save_prefix='', saveKey=''):
    # チェスボードと画像上のコーナーの座標の記録
    obj_points_all = []
    img_points_all = []
    isMask = False

    #-----------------
    # チェスボード（X,Y,Z=0）のコーナーの座標を計算（xy座標は左上原点、z軸は0にあると仮定）
    obj_points = np.zeros( (np.prod(corner_size), 3), np.float32 )
    obj_points_tmp = np.indices(corner_size)
    obj_points[:,:2] = obj_points_tmp.T.reshape(-1, 2)
    obj_points *= square_size
    #-----------------
    
    #-----------------
    # 画像を撮影し、コーナーの検出
    if isVideo:
        capture = cv2.VideoCapture(cameraID)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, img_w)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, img_h)

    cnt = 0

    while cnt < img_num:
        # 画像の取得
        if isVideo:
            ret, img = capture.read()
        else:
            img = cv2.imread(f"{imgPath}/{load_prefix}_captured_{cnt}.png")

        # マスクをかける
        if len(mask_imgs):
            isMask = True
            img_mask = copy.deepcopy(img)
            img_mask = img_mask*mask_imgs[cnt]
            img_mask = img_mask.astype(np.uint8)

            # グレーコードに変換
            gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
        else:
            # グレーコードに変換
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # チェスボードのコーナーを検出
        c_flag, corners = cv2.findChessboardCorners(gray, corner_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

        #-----------------
        # チェスボードのコーナーが検出された場合
        if c_flag == True:            
            # コーナーの高精度化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30 , 0.01)
            corners = cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)

            # コーナーの描画
            img_corner = copy.deepcopy(img)
            cv2.drawChessboardCorners(img_corner,corner_size,corners,c_flag)

            if len(saveKey) > 0:
                if cv2.waitKey(200) & 0xFF == ord('s'):                
                    print(f"{cnt+1}/{img_num}")
            else:                
                print(f"{cnt+1}/{img_num}")

            # コーナー点の記録
            img_points_all.append(corners.reshape(-1, 2))   # 画像座標系
            obj_points_all.append(obj_points)

            if isVideo:
                cv2.imwrite(f"{imgPath}/{save_prefix}_captured_{cnt}.png",img)

                if isMask:
                    cv2.imwrite(f"{imgPath}/{save_prefix}_captured_mask_{cnt}.png",img_mask)
            else:                        
                cv2.imwrite(f"{imgPath}/{save_prefix}_detected_corner_{cnt}.png",img_corner)
                if isMask:
                    cv2.imwrite(f"{imgPath}/{save_prefix}_captured_mask_{cnt}.png",img_mask)
            cnt += 1

            img = img_corner
        elif not isVideo:
            print(f"{cnt+1}/{img_num}")

            # コーナー点の記録
            img_points_all.append([])   # 画像座標系
            obj_points_all.append(obj_points)     

            cnt += 1
        #-----------------

        # 描画
        cv2.imshow('image', img)

        if isMask:
            cv2.imshow('image_mask', img_mask)
        
        # 200ms待つ
        if cv2.waitKey(waitTime) & 0xFF == ord('q'):
            break
    #-----------------

    # ウィンドウを閉じる
    cv2.destroyAllWindows()

    return img_points_all, obj_points_all, gray
#-----------------    

#-----------------    
# チェスボードの撮影とキャリブレーション
def calibration(cameraID=1, waitTime=1000, square_size=2.4, corner_size=(7,10), img_num=40, img_w=1280, img_h=720, imgPath='images', isVideo=True, prefix='', saveKey=''):
    # チェスボードと画像上のコーナーの座標の記録
    obj_points_all = []

    # 画像からコーナーの検出（画像座標系のコーナー点をimg_points_allに記録）
    img_points_all, obj_points_all, gray = detectCornerFromImages(cameraID, waitTime, corner_size, img_num, img_w=img_w, img_h=img_h, imgPath=imgPath, isVideo=isVideo, load_prefix=prefix, save_prefix=prefix, saveKey=saveKey)

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

#-----------------
def calcPoseAndPosition(img_points, obj_points, mtx, dist, load_img_path, save_img_path, waitTime=1000):
    # PnP問題を解き、物体座標系からカメラ画像平面への射影行列（回転と並進）を求める
    _, rvec, tvec = cv2.solvePnP(obj_points, img_points, mtx, dist)

    # 回転ベクトル（回転行列のコンパクト表現）
    print(f"rotation:\n{rvec}")

    # 並進ベクトル
    print(f"travel:\n{tvec}")

    # 物体座標系の軸の設定
    obj_origin = np.float32([[5.0,0,0], [0,10,0], [0,0,-20]])

    #-----------------
    # 物体座標の原点を画像座標に射影

    # 画像の読み込み
    img = cv2.imread(load_img_path)

    obj_origin_img, jac = cv2.projectPoints(obj_origin, rvec, tvec, mtx, dist)
    img = draw_origin(img,tuple(img_points[0]),obj_origin_img)
    cv2.imwrite(save_img_path,img)
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
    camera_points = np.dot(R,obj_points.T)+tvec
    #-----------------

    # waitTime[ms]待つ
    cv2.waitKey(waitTime) & 0xFF == ord('c')

    # ウィンドウを閉じる
    cv2.destroyAllWindows()

    return rvec, tvec, obj_origin_camera, camera_points
#-----------------
#===================

#===================
# main

if __name__ == '__main__':

    # カメラID
    cameraID = 0

    # -1: no calibration, 1: calibration from camera, 2: calibration from files
    calibration_mode = -1

    # -1: no demonstration, 1: demonstration from camera, 2: demonstration from files
    demo_mode = 1

    # パラメータの設定
    #square_size = 2.3      # 正方形の1辺のサイズ[cm]
    square_size = 3.3      # 正方形の1辺のサイズ[cm]    
    #corner_size = (10, 7)  # 交差ポイントの数
    corner_size = (4, 3)  # 交差ポイントの数
    img_num = 100 # 参照画像の枚数
    img_h = 1080 #720
    img_w = 1920 #1280

    if calibration_mode == 1:
        # from video
        calibration(cameraID=cameraID, square_size=square_size, corner_size=corner_size, img_num=img_num, img_w=img_w, img_h=img_h, imgPath='images', prefix='normal')
    elif calibration_mode == 2:
        # from images
        calibration(square_size=square_size, corner_size=corner_size, img_num=img_num, imgPath='images', isVideo=False, prefix='normal')

    #-----------------
    # カメラ行列と歪みパラメータの読み込み
    with open('normal_camera_param.pkl','rb') as fp:
        mtx = pickle.load(fp) # カメラ行列
        dist = pickle.load(fp) # 歪みパラメータ
        _ = pickle.load(fp) # RMS誤差
        img_points_all = pickle.load(fp) # 画像のコーナー点
        obj_points_all = pickle.load(fp) # 物体のコーナー点（z=0）
    #-----------------

    if demo_mode == 1:
        img_num = 40
        isVideo = False
        corner_size = (4, 3)  # 交差ポイントの数
        corner_num = np.product(corner_size)

        #-----------------
        # 本物のコーナー検出

        # 鏡のマスク画像の作成（鏡は画像の中央下にあると仮定）
        mask_imgs = [np.ones([img_h,img_w,3]) for i in range(img_num)]
        for i in range(img_num):
            mask_imgs[i][int(img_h-img_h/2):img_h,int(img_w/2-img_w/6):int(img_w/2+img_w/6),:] = 0

        # コーナー検出
        img_points_all, obj_points_all, _ = detectCornerFromImages(cameraID=cameraID, corner_size=corner_size, img_num=img_num, img_w=img_w, img_h=img_h, mask_imgs=mask_imgs, isVideo=isVideo, imgPath='images', load_prefix='demo', save_prefix='demo')

        # カメラ座標の取得
        camera_points_all = []
        for j in range(img_num):
            load_img_path = f"images/demo_captured_{j}.png"
            save_img_path = f"images/demo_obj_axis_{j}.png"   
            rvec, tvec, obj_origin_camera, camera_points = calcPoseAndPosition(img_points_all[j], obj_points_all[j], mtx=mtx, dist=dist, load_img_path=load_img_path, save_img_path=save_img_path)
            camera_points_all.append(camera_points)
        #-----------------
        
        #-----------------
        # 鏡像のコーナー検出

        # チェスボードのマスク画像の作成
        mask_imgs = [np.ones([img_h,img_w,3]) for i in range(img_num)]
        for i in range(img_num):
            mins=np.min(img_points_all[i],axis=0)
            maxs=np.max(img_points_all[i],axis=0)
            mask_imgs[i][int(mins[1]):int(maxs[1]),int(mins[0]):int(maxs[0]),:] = 0

        # コーナー検出
        img_mirror_points_all, _, _ = detectCornerFromImages(cameraID=cameraID, corner_size=corner_size, img_num=img_num, img_w=img_w, img_h=img_h, mask_imgs=mask_imgs, isVideo=False, imgPath='images', load_prefix='demo', save_prefix='demo_mirror')
        detected_inds = np.where(np.array([len(img_mirror_points_all[i]) for i in range(len(img_mirror_points_all))])>0)[0]
        #-----------------

        #-----------------
        # 反射ベクトルの類似度スコアの計算
        scores = []
        fx = mtx[0,0]
        fy = mtx[1,1]
        cx = mtx[0,2]
        cy = mtx[1,2]
        for j in range(len(detected_inds)):
            pdb.set_trace()

            # 鏡像のコーナー点を、corner_sizeの形に合わせて並べ替え
            img_mirror_points=np.reshape(np.transpose(np.expand_dims(img_mirror_points_all[detected_inds[j]],-1),[1,0,2]),[2,corner_size[1],corner_size[0]])

            # 鏡像なので左右反転
            img_mirror_points = img_mirror_points[:,:,::-1]

            # 鏡像のコーナー点を本物のコーナー点に合わせて並べ替え
            img_mirror_points = np.reshape(img_mirror_points,[2,corner_num])

            # 鏡像のコーナー点のカメラ座標系のベクトルを求める
            v = np.vstack([img_mirror_points[0]-cx,img_mirror_points[1]-cy,np.ones(corner_num)*(fx+fy)/2])
            v /= np.linalg.norm(v,axis=0)

            # 本物のコーナー点のカメラ座標
            Pc = camera_points_all[detected_inds[j]]
            p = Pc/np.linalg.norm(Pc,axis=0)

            # 鏡像コーナーベクトルvと本物コーナーベクトルpの外積と局所球の中心ベクトルsは直交関係にあるので、内積が0
            Q = np.array([np.cross(v[:,i],p[:,i]) for i in range(corner_num)])
            U, S, V = np.linalg.svd(Q)
            s = V[[-1],:].T

            # 局所球の半径rと距離dの候補
            R, D = np.meshgrid(np.linspace(1,50,100),np.linspace(10,100,100))

            # 半径rと距離dとv.Tsの関係に基づき候補の有効性を確認
            term = 1-(R/D)**2
            vs = np.dot(v.T,s)
            val = (term>0) & (term<=np.min(vs**2))

            # v.ts, validsの行列版
            VStile = np.tile(np.expand_dims(vs,-1),[R.shape[0],R.shape[1]])

            # Kの計算
            K=(D*VStile - np.sqrt(D**2*(VStile**2-1) + R**2))

            # Mcの計算
            Ktile = np.tile(np.expand_dims(K,0),[3,1,1,1])
            Vtile = np.tile(np.expand_dims(np.expand_dims(v,-1),-1),[1,1,R.shape[0],R.shape[1]])
            Mctile = Vtile*Ktile

            # Pc-Mcの計算
            Pctile = np.tile(np.expand_dims(np.expand_dims(Pc,-1),-1),[1,1,R.shape[0],R.shape[1]])
            PcMinusMc = Pctile-Mctile
            PcMinusMcNorm = PcMinusMc/np.linalg.norm(PcMinusMc,axis=0) 

            # 法線ベクトルnの計算
            Stile = np.tile(np.expand_dims(np.expand_dims(s,-1),-1),[1,len(vs),R.shape[0],R.shape[1]])
            Dtile = np.tile(np.expand_dims(np.expand_dims(D,0),0),[3,len(vs),1,1])
            Rtile = np.tile(np.expand_dims(np.expand_dims(R,0),0),[3,len(vs),1,1])
            Sctile = Stile*Dtile
            Ntile = (Mctile-Sctile)/Rtile

            # 反射ベクトルVVの計算、PcMinusMcNormとの差を計算
            VV = Vtile - 2*Ntile*np.sum(Ntile*Vtile,axis=0,keepdims=True)
            score = np.mean(np.sum(VV*PcMinusMcNorm,axis=0),axis=0)
            score[val==False] = -1
            scores.append(score)
            scoreMax = np.max(score)
            IndMax = np.where(score==scoreMax)
            Rmax = R[IndMax][0]
            Dmax = D[IndMax][0]

            # カメラ座標と画像座標の計算
            camera_mirror_points = np.tile(K[:,IndMax[0],IndMax[1]],[1,3])*v.T
            ux = fx*camera_mirror_points[:,0]/camera_mirror_points[:,2]+cx
            uy = fy*camera_mirror_points[:,1]/camera_mirror_points[:,2]+cy

            load_img_path = f"images/demo_captured_{detected_inds[j]}.png"
            save_img_path = f"images/demo_mirror_estimated_corner_{detected_inds[j]}.png"
            img = cv2.imread(load_img_path)
            for i in range(len(ux)):
                img = cv2.circle(img,(int(ux[i]),int(uy[i])),5,(0,0,255),thickness=-1)
            cv2.imwrite(save_img_path,img)

            # plot
            print(f"score@(R,D)={scoreMax}({Rmax},{Dmax})")          
            fig,ax=plt.subplots(1,1)
            hmap = ax.pcolormesh(R,D,score,cmap="coolwarm")            
            fig.colorbar(hmap,ax=ax,orientation="vertical")
            plt.plot(Rmax,Dmax,'ro',markersize=12)
            ax.set_xlabel('Radius, r [cm]')
            ax.set_ylabel('Distance, d [cm]')
            plt.savefig(f'images/score_{detected_inds[j]}.png')

        pdb.set_trace()
        #-----------------            


    elif demo_mode == 2:
        #===================
        # PnP問題を解き、物体（チェッカーボード）の原点を画像座標に射影
        # j: 鏡のindex
        camera_points_all = []
        for j in range(len(img_points_all)):
            load_img_path = f"images/normal_captured_{j}.png"
            save_img_path = f"images/normal_obj_axis_{j}.png"
            rvec, tvec, obj_origin_camera, camera_points = calcPoseAndPosition(img_points_all[j], obj_points_all[j], mtx=mtx, dist=dist, load_img_path=load_img_path, save_img_path=save_img_path)
            camera_points_all.append(camera_points)

        #-----------------
        # カメラ座標系におけるコーナー点のプロット
        camera_points_all = np.array(camera_points_all)    
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(camera_points_all[0][0],camera_points_all[0][1],camera_points_all[0][2],color="red")
        ax.scatter(camera_points_all[1][0],camera_points_all[1][1],camera_points_all[1][2],color="blue")
        ax.scatter(camera_points_all[2][0],camera_points_all[2][1],camera_points_all[2][2],color="green")
        ax.scatter(camera_points_all[3][0],camera_points_all[3][1],camera_points_all[3][2],color="cyan")
        ax.scatter(camera_points_all[4][0],camera_points_all[4][1],camera_points_all[4][2],color="magenta")
        ax.scatter(0,0,0,color="black",s=50)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend(['j=0','j=1','j=2','j=3','j=4'])

        plt.show()
        #-----------------       

        pdb.set_trace()
    #===================
