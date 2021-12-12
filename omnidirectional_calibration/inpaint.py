import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

input_path = "./complement_image/input/"
output_path = "./complement_image/output/"
img_no = "cut63.png"

img = cv2.imread(os.path.join(input_path) + os.path.join(img_no))
h, w = img.shape[:2]
print(img)

mask = np.zeros((h, w), dtype=np.uint8)
#円形のマスク画像作成
#radius = h//2-160：円の半径が大きいと補完画像の下にエッジが発生するため円の半径を少し小さめに設定
cv2.circle(mask,center=(h//2,w//2),radius=h//2-160,color=255,thickness=-1)
#非ゼロの領域が補完されるためマスク画像の領域を反転
mask = 255-mask 

#inpaint
dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

cv2.imwrite(os.path.join(output_path) + os.path.join(img_no), dst)