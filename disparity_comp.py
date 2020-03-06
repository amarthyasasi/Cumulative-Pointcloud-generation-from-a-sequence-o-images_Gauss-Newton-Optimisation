import numpy as np
import cv2
import open3d as o3d

left = cv2.imread('left.png')
right = cv2.imread('right.png')

window_size = 5
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,             # max_disp has to be dividable by 16 f. E. HH 192, 256
    blockSize=3,
    P1=4  * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    P2=32  * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63)
      
numDisparities=144
minDisparity=-39
disparity = stereo.compute(right,left).astype('float32')
disparity = (disparity - minDisparity) / numDisparities
print(np.min(disparity),np.max(disparity))
cv2.imshow('Disparity Map', disparity)
cv2.waitKey(3000)

######-----STEREO BM------########
# left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
# right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
# bm_matcher = cv2.StereoBM_create(numDisparities=160, blockSize=5)
# bm_matcher.setTextureThreshold(5)
# bm_matcher.setMinDisparity(-10)
# bm_matcher.setSpeckleWindowSize(100)
# bm_matcher.setSpeckleRange(64)
# bm_matcher.setUniquenessRatio(0)
# bm_matcher.setPreFilterSize(31)
# bm_matcher.setPreFilterCap(31)
# disp_bm = bm_matcher.compute(left, right)
# cv2.imshow('Disparity Map with BM', disp_bm)
# cv2.waitKey(8000)