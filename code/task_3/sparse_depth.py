import numpy as np
import cv2
import glob
from numpy import loadtxt
import argparse
from matplotlib import pyplot as plt




cmatrix_left = np.loadtxt("D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/Camera_Matrix_Left.txt")
cmatrix_right = np.loadtxt("D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/Camera_Matrix_Right.txt")
dis_coeff_left = np.loadtxt("D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/Distortion_Coeffs_Left.txt")
dis_coeff_right = np.loadtxt("D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/Distortion_Coeffs_Right.txt")
P1 = np.loadtxt("D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/P1.txt")
P2 = np.loadtxt("D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/P2.txt")
img_l=cv2.imread('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/images/task_3_and_4/left_0.png')

img_r=cv2.imread('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/images/task_3_and_4/right_0.png')

grayl= cv2.cvtColor(img_l,cv2.COLOR_BGR2GRAY)
grayr= cv2.cvtColor(img_r,cv2.COLOR_BGR2GRAY)

height,width = img_l.shape[:2]

mapx1, mapy1 = cv2.initUndistortRectifyMap(cmatrix_left, dis_coeff_left, None, cmatrix_left, (width,height), 5)
undis_l = cv2.remap(grayl,mapx1, mapy1, cv2.INTER_LINEAR)

mapx2, mapy2 = cv2.initUndistortRectifyMap(cmatrix_right, dis_coeff_right, None, cmatrix_right, (width,height), 5)
undis_r = cv2.remap(grayr,mapx2, mapy2, cv2.INTER_LINEAR)

# cv2.imshow('img_l',undis_l)
# cv2.waitKey(1000)
# cv2.imshow('img_r',undis_r)
# cv2.waitKey(3000)

orb=cv2.ORB_create()

feature_l= orb.detect(undis_l,None)
feature_l,descriptor_l=orb.compute(undis_l,feature_l)
new_l=cv2.drawKeypoints(undis_l,feature_l,np.array([]),(0,250,0),0)

cv2.imshow('img_l',new_l)
cv2.waitKey(2000)
cv2.imwrite('featurel.png',new_l)

feature_r= orb.detect(undis_r,None)
feature_r,descriptor_r=orb.compute(undis_r,feature_r)
new_r=cv2.drawKeypoints(undis_r,feature_r,np.array([]),(0,250,0),0)

cv2.imshow('img_r',new_r)
cv2.imwrite('featurer.png',new_r)
cv2.waitKey(2000)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(descriptor_l,descriptor_r)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
match_img = cv2.drawMatches(grayl,feature_l,grayr,feature_r,matches[:40],None,flags=2)

plt.imshow(match_img),plt.show()

# Initialize lists
list_left = []
list_right= []

# For each match...
for mat in matches:

    # Get the matching keypoints for each of the images

    img1_idx = mat.queryIdx
    img2_idx = mat.trainIdx

    # x - columns
    # y - rows
    # Get the coordinates
    (x1, y1) = feature_l[img1_idx].pt
    (x2, y2) = feature_r[img2_idx].pt

    # Append to each list
    list_left.append((x1, y1))
    list_right.append((x2, y2))
list_left=np.array(list_left)
list_right=np.array(list_right)


tp=cv2.triangulatePoints(P1,P2,list_left.T,list_right.T)

print(tp.shape)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax = plt.axes(projection='3d')
ax.scatter3D(tp[0]/tp[3], tp[1]/tp[3], tp[2]/tp[3],c='b', marker='o')
plt.show()
