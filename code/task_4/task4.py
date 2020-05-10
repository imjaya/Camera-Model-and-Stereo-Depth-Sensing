import argparse
import sys
import os
import glob
import numpy as np
#from sklearn.preprocessing import normalize
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
cmatrix_left = np.loadtxt("D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/Camera_Matrix_Left.txt")
cmatrix_right = np.loadtxt("D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/Camera_Matrix_Right.txt")
dis_coeff_left = np.loadtxt("D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/Distortion_Coeffs_Left.txt")
dis_coeff_right = np.loadtxt("D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/Distortion_Coeffs_Right.txt")

R1=np.loadtxt("D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/R1.txt")
R2=np.loadtxt("D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/R2.txt")
P1=np.loadtxt("D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/P1.txt")
P2=np.loadtxt("D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/P2.txt")
Q=np.loadtxt("D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/Q.txt")



rows = 6
cols = 9
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

objpoints = []  # 3d point in real world space
imgpoints_l = []  # 2d points in image plane.
imgpoints_r = []  # 2d points in image plane.



imgL = cv2.imread('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/images/task_3_and_4/left_7.png')  # downscale images for faster processing if you like
imgR = cv2.imread('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/images/task_3_and_4/right_7.png')
height,width = imgL.shape[:2]

mapx1, mapy1 = cv2.initUndistortRectifyMap(cmatrix_left, dis_coeff_left, R1, cmatrix_left, (width,height), 5)
rect_l = cv2.remap(imgL,mapx1, mapy1, cv2.INTER_LINEAR)

mapx2, mapy2 = cv2.initUndistortRectifyMap(cmatrix_right, dis_coeff_right,R2, cmatrix_right, (width,height), 5)
rect_r = cv2.remap(imgR,mapx2, mapy2, cv2.INTER_LINEAR)

cv2.imshow('img_l',rect_l)
cv2.waitKey(1000)
cv2.imshow('img_r',rect_r)
cv2.waitKey(3000)

# SGBM Parameters -----------------
window_size = 3                     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

left_matcher = cv2.StereoSGBM_create( minDisparity=0, numDisparities=160,blockSize=5, P1=8 * 3 * window_size ** 2,P2=32 * 3 * window_size ** 2,disp12MaxDiff=1,uniquenessRatio=15,speckleWindowSize=0,speckleRange=2, preFilterCap=63,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)


# FILTER Parameters
lmbda = 10000
sigma = 1.2
visual_multiplier = 0.5

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)
displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
displ = np.int16(displ)
dispr = np.int16(dispr)
filteredImg = wls_filter.filter(displ, imgL, None, dispr)

filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
filteredImg = np.uint8(filteredImg)
cv2.imshow('Disparity Map', filteredImg)
cv2.waitKey()
cv2.destroyAllWindows()

v=np.empty([480,640,3])
v=cv2.reprojectImageTo3D(displ,Q)
print(v)
