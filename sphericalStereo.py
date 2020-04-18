import numpy as np
import cv2
from matplotlib import pyplot as plt
import pdb

# ==================
# functions
class sphericalStereo:
    resultPath = "result"
    
    #-----------------
    # コンストラクタ
    def __init__(self,imgPathL,imgPathR):
        # 画像の読み込み
        imgL = cv2.imread(imgPathL) 
        imgR = cv2.imread(imgPathR) 
        self.imgL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY) 
        self.imgR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY) 

        # 前後の画像に分割
        width = int(self.imgL.shape[1]/2)
        self.imgLFront = cv2.rotate(self.imgL[:,:width], cv2.ROTATE_90_CLOCKWISE)
        self.imgLBack = cv2.rotate(self.imgL[:,width:], cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.imgRFront = cv2.rotate(self.imgR[:,:width], cv2.ROTATE_90_CLOCKWISE)
        self.imgRBack = cv2.rotate(self.imgR[:,width:], cv2.ROTATE_90_COUNTERCLOCKWISE)               
    #-----------------
            
    #-----------------
    # チェスボードのコーナー検出
    def findCorners(self,img,colNum=10,rowNum=7,postFix=""):
        # チェスボードの検出
        c_flag, corners = cv2.findChessboardCorners(img, (colNum, rowNum))

        if c_flag:
            # コーナー一の高精度化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30 , 0.001)
            corners = cv2.cornerSubPix(img,corners,(11,11),(-1,-1),criteria)

            # グレースケールからカラーに変更
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

            # コーナーの描画
            cv2.drawChessboardCorners(img,(colNum,rowNum),corners,c_flag)        
            cv2.imwrite(f"{self.resultPath}/detected_corner_{postFix}.png",img)

        return corners
    #-----------------

    #-----------------
    # 特徴点のマッチング
    def featureMatching(self,imgL,imgR,k=2, postFix=""):
        # 特徴抽出（SIFT特徴）
        detector = cv2.xfeatures2d.SURF_create()
        #detector = cv2.xfeatures2d.SIFT_create()

        # キーポイントと記述の取得
        kp1, des1 = detector.detectAndCompute(imgL,None)
        kp2, des2 = detector.detectAndCompute(imgR,None)

        # knnマッチング
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)

        good = []
        match_param = 0.6
        for ind in range(len(matches)):
            m = matches[ind][0]
            n = matches[ind][1]
            if m.distance < match_param*n.distance:
                good.append([m])

        # 特徴点マッチングの可視化
        img = cv2.drawMatchesKnn(imgL,kp1,imgR,kp2,good,None,flags=2)
        cv2.imwrite(f"{self.resultPath}/matching_result_{postFix}.png",img)
    #-----------------
# ==================

# ==================
# Main
mySS = sphericalStereo("images/R0020021.JPG","images/R0020022.JPG")

#-----------------
# チェスボードのコーナー検出の例
cornerLBack = mySS.findCorners(mySS.imgLBack,postFix="LBack")
cornerRBack = mySS.findCorners(mySS.imgRBack,postFix="RBack")
#-----------------

pdb.set_trace()

#-----------------
# 画像特徴マッチングの例
mySS.featureMatching(mySS.imgLFront,mySS.imgRFront,postFix="Front")
mySS.featureMatching(mySS.imgLBack,mySS.imgRBack,postFix="Back")
#-----------------

# ==================

