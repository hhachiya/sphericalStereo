import numpy as np
import cv2
from matplotlib import pyplot as plt
import pdb
import pathlib
import pickle
import copy
from scipy.spatial.transform import Rotation
import numpy as np
import copy
from mpl_toolkits.mplot3d import Axes3D


def calibration(square_size=38,corner_size=(4,3),img_num=20):
    xmlpath = "omniCalibrate26.xml"
    
    
    fs = cv2.FileStorage(xmlpath,cv2.FileStorage_READ)
    Rot = fs.getNode("R").mat()
    T = fs.getNode("T").mat()
    idx = fs.getNode("idx").mat()
    
    #コーナー数　12 = 3*4
    v = np.empty((12,3))
    p = np.empty((12,3))
    obj_points = np.zeros( (np.prod(corner_size), 3), np.float32 )
    obj_points_tmp = np.indices(corner_size)
    obj_points[:,:2] = obj_points_tmp.T.reshape(-1, 2)
    obj_points *= square_size

    #x反転
    for i in range(12):
        obj_points[i,0] = square_size*3 - obj_points[i,0]

    for i in range(idx.shape[1]):
        if idx[0,i] > 62:
            #pdb.set_trace()
            #前(鏡像)の画像    
            img = cv2.imread(f"stereo26/LBack26/cut{idx[0,i]:02d}.png")
            #鏡像の範囲指定
            #img2[縦、横]
            mirror_x = 1450
            mirror_y = 1550
            mirror_w = 724
            img2 = img[mirror_y:mirror_y+mirror_w,mirror_x:mirror_x+mirror_w]
            #img2拡大
            img3 = cv2.resize(img2,(img.shape[0],img.shape[1]))
            #cv2.imwrite("stereo26/result/cut{idx[0,i]:02d}.png",img3)
            #コーナー検出
            
            c_flag, corners = cv2.findChessboardCorners(img3, corner_size)
            if c_flag == True:
                print("found")
                
                #元の座標に戻す
                cornersx = (corners[0:,0,0])/4 + mirror_x
                cornersy = corners[0:,0,1]/4 + mirror_y
                corners2 = np.stack([cornersx,cornersy],1)

                #可視化
                img_tmp = copy.deepcopy(img3)
                cv2.drawChessboardCorners(img_tmp,corner_size,corners,c_flag)
                
                #カメラから鏡像へのレイ
                #画像中心1448
                #画角180度のピクセル1270 
                cx = 1448
                cy = 1448
                pix180 = 1270
                for j in range(12):
                    alpha = np.arctan2(abs(cy-corners2[j][1]),abs(cx-corners2[j][0]))
                    theta = (np.pi / 2) * (np.sqrt(np.power(cx-corners2[j][0],2) + np.power(cy-corners2[j][1],2))/pix180)
                    v[j] = np.array([[np.cos(alpha)*np.sin(theta),np.sin(theta)*np.sin(alpha),np.cos(theta)]])

                    if cornersy[j] > img.shape[1]/2:
                        v[j][1] = - v[j][1]
                
                #コーナーの3次元座標
                #後ろの画像(チェスボード)
                rotvec2 = Rot[i,0]
                tvec = T[i,0]
                rot2 = Rotation.from_rotvec(rotvec2)
                dcm2 = rot2.as_dcm()
                
                #座標系を反転して前の画像のカメラ座標系と一致させる(yは上向きにする)
                for k in range(12):
                    p[k] = -(np.dot(dcm2,obj_points[k].T) + tvec)
                #print("p1:",p)
                

                #print(v)

                pn = p.T/np.linalg.norm(p.T,axis=0)
                vn = v.T/np.linalg.norm(v.T,axis=0)

                Q = np.array([np.cross(vn[:,l],pn[:,l]) for l in range(12)])
                U, S, V = np.linalg.svd(Q)
                s = V[[-1],:].T

                #print('V',V)
                #print('S',S)
                #print('s',s)
                
                R,D = np.meshgrid(np.linspace(10,500,100),np.linspace(100,1000,100))
                
                #print('R',R)
                #print('D',D)
                #print('D',D.shape)
                term = 1-(R/D)**2
                vs = np.dot(vn.T,s)
                val = (term>0)&(term<=np.min(vs**2))

                VStile = np.tile(np.expand_dims(vs,-1),[R.shape[0],R.shape[1]])
                #print(vs.shape)
                #print(VStile.shape)
                K = (D*VStile - np.sqrt(D**2*(VStile**2-1) + R**2))
                #print("K",K.shape)
                Ktile = np.tile(np.expand_dims(K,0),[3,1,1,1])
                #print("Kt",Ktile.shape)
                Vtile = np.tile(np.expand_dims(np.expand_dims(vn,-1),-1),[1,1,R.shape[0],R.shape[1]])
                Mctile = Vtile*Ktile


                Ptile = np.tile(np.expand_dims(np.expand_dims(p.T,-1),-1),[1,1,R.shape[0],R.shape[1]])
                PcMinusMc = Ptile-Mctile
                PcMinusMcNorm = PcMinusMc/np.linalg.norm(PcMinusMc,axis=0)

                Stile = np.tile(np.expand_dims(np.expand_dims(s,-1),-1),[1,len(vs),R.shape[0],R.shape[1]])
                Dtile = np.tile(np.expand_dims(np.expand_dims(D,0),0),[3,len(vs),1,1])
                Rtile = np.tile(np.expand_dims(np.expand_dims(R,0),0),[3,len(vs),1,1])
                Sctile = Stile*Dtile
                Ntile = (Mctile-Sctile)/Rtile

                # 反射ベクトルVVの計算、PcMinusMcNormとの差を計算
                VV = Vtile - 2*Ntile*np.sum(Ntile*Vtile,axis=0,keepdims=True)
                VV = VV/np.linalg.norm(VV,axis = 0)
                #print("VV",VV.shape)
                #print("Pc",PcMinusMcNorm.shape)
                score = np.mean(np.sum(VV*PcMinusMcNorm,axis=0),axis=0)

                score[val==False] = -1
                #print(score.shape)
                #np.where(score < 0.99,0,-1)
                scores = []
                scores.append(score)
                scoreMax = np.max(score)
                IndMax = np.where(score==scoreMax)
                #print("IndR")
                Rmax = R[IndMax][0]
                Dmax = D[IndMax][0]

                # plot
                
                print(f"score@(R,D)={scoreMax}({Rmax},{Dmax})")          
                fig,ax=plt.subplots(1,1)
                hmap = ax.pcolormesh(R,D,score,vmin=-1,vmax=1, cmap="coolwarm")            
                fig.colorbar(hmap,ax=ax,orientation="vertical")
                plt.plot(Rmax,Dmax,'ro',color = (0,1,0) ,markersize=12)
                ax.set_xlabel('Radius, r [mm]',fontsize=18)
                ax.set_ylabel('Distance, d [mm]',fontsize=18)
                plt.savefig(f'stereo26/result/score{idx[0,i]:02d}.pdf')

            else:
                print("not found")    
    


if __name__ == '__main__':
    calibration()
 
