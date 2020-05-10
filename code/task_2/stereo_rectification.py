import numpy as np
import cv2
import glob
from numpy import loadtxt
import argparse

cmatrix_left = np.loadtxt("D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/Camera_Matrix_Left.txt")
cmatrix_right = np.loadtxt("D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/Camera_Matrix_Right.txt")
dis_coeff_left = np.loadtxt("D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/Distortion_Coeffs_Left.txt")
dis_coeff_right = np.loadtxt("D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/Distortion_Coeffs_Right.txt")


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints_left = []
imgpoints_right = [] # 2d points in image plane.



img1 = cv2.imread('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/images/task_2/left_0.png')
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

# Find the chess board corners
ret1, corners1 = cv2.findChessboardCorners(gray1, (6,9),None)
objpoints.append(objp)

# If found, add object points, image points (after refining them)
if ret1 == True:
    imgpoints_left.append(corners1)
    img1 = cv2.drawChessboardCorners(img1, (6,9), corners1,ret1)



img2= cv2.imread('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/images/task_2/right_0.png')
gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# Find the chess board corners
ret2, corners2 = cv2.findChessboardCorners(gray2, (6,9),None)

# If found, add object points, image points (after refining them)
if ret2 == True:
    imgpoints_right.append(corners2)
    img2 = cv2.drawChessboardCorners(img2, (6,9), corners2,ret2)



height,width = img2.shape[:2]




#retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F=cv2.stereoCalibrate(objpoints,imgpoints_left,imgpoints_right,cmatrix_left,dis_coeff_left,cmatrix_right,dis_coeff_right,(width,height),R,T,criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 1e-5),flags=cv2.CALIB_FIX_INTRINSIC)
T = np.zeros((3, 1), dtype=np.float64)
R = np.eye(3, dtype=np.float64)



criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 1e-5)

retval, cmatrix_left, dis_coeff_left, cmatrix_right, dis_coeff_right, R, T, E, F=cv2.stereoCalibrate\
    (objpoints,imgpoints_left,imgpoints_right,cmatrix_left,dis_coeff_left,cmatrix_right,dis_coeff_right,\
        (width,height),R,T,criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 1e-5),flags=cv2.CALIB_FIX_INTRINSIC)

np.savetxt('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/E.txt',E)
np.savetxt('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/R.txt',R)
np.savetxt('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/T.txt',T)
np.savetxt('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/F.txt',F)
img_left=np.array(imgpoints_left)
img_right=np.array(imgpoints_right)
img_left.shape = (54, 1, 2)
img_right.shape = (54, 1, 2)

img_left = cv2.undistortPoints(img_left,cmatrix_left,dis_coeff_left)

img_right = cv2.undistortPoints(img_right,cmatrix_right,dis_coeff_right)






I=np.eye(3, dtype=np.float64)
O=np.zeros((3, 1), dtype=np.float64)

P1=np.hstack((I,O))
P2=np.hstack((R,T))

np.savetxt('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/P1.txt',P1)
np.savetxt('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/P2.txt',P2)


final_points=cv2.triangulatePoints(P1,P2,img_left,img_right)
tp=final_points

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection



fig = plt.figure()
plt.axis('equal')
#ax = fig.add_subplot(111, projection='3d')
ax = plt.axes(projection='3d')
plt.title('Before Rectification')

# vertices of a pyramid
v = np.array([[-1, -1, 1], [1, -1, 1], [1, 1, 1],  [-1, 1, 1], [0, 0, 0]])
ax.scatter3D(v[:, 0], v[:, 1], v[:, 2])

# generate list of sides' polygons of our pyramid
verts = [ [v[0],v[1],v[4]], [v[0],v[3],v[4]],
 [v[2],v[1],v[4]], [v[2],v[3],v[4]], [v[0],v[1],v[2],v[3]]]

# plot sides
ax.add_collection3d(Poly3DCollection(verts,
 facecolors='white', linewidths=1, edgecolors='k', alpha=.25))


v2 = np.array([[1.4264, -0.9801, 0.9212], [3.4254, -0.9930, 0.9828], [3.4380, 1.0069, 0.9946],  [1.4390, 1.0198, 0.9330], [2.4630, 0.0191, -0.0416]])
ax.scatter3D(v2[:, 0], v2[:, 1], v2[:, 2])

verts2 = [ [v2[0],v2[1],v2[4]], [v2[0],v2[3],v2[4]],
 [v2[2],v2[1],v2[4]], [v2[2],v2[3],v2[4]], [v2[0],v2[1],v2[2],v2[3]]]

ax.add_collection3d(Poly3DCollection(verts2,
 facecolors='white', linewidths=1, edgecolors='k', alpha=.25))





x=tp[0]/tp[3]
y=tp[1]/tp[3]
z=tp[2]/tp[3]
ax.scatter3D(tp[0]/tp[3], tp[1]/tp[3], tp[2]/tp[3],c=tp[2]/tp[3], marker='o')

ax.set_xlim(-10, 10)
ax.set_ylim(-10,10)
ax.set_zlim(-10,10)


plt.show()



fig = plt.figure()
plt.axis('equal')
#ax = fig.add_subplot(111, projection='3d')
ax = plt.axes(projection='3d')
plt.title('Rectified')


# vertices of a pyramid
v3 = np.array([[-1.0245, -0.9893, 0.9858], [0.9752, -1.0047, 1.0196], [0.9907, 0.9952, 1.0140],  [-1.0090, 1.0106, 0.9801], [0, 0, 0]])
ax.scatter3D(v3[:, 0], v3[:, 1], v3[:, 2])



# generate list of sides' polygons of our pyramid
verts3 = [ [v3[0],v3[1],v3[4]], [v3[0],v3[3],v3[4]],
 [v3[2],v3[1],v3[4]], [v3[2],v3[3],v3[4]], [v3[0],v3[1],v3[2],v3[3]]]

# plot sides
ax.add_collection3d(Poly3DCollection(verts3,
 facecolors='white', linewidths=1, edgecolors='k', alpha=.25))


v4 = np.array([[1.3670, -1.0024 , 0.9850], [3.3604, -1.0436, 1.1418], [3.4003, 0.9559, 1.1607],  [1.4069, 0.9971, 1.0039], [2.4622, -0.0154, 0.0760]])
ax.scatter3D(v4[:, 0], v4[:, 1], v4[:, 2])




verts4 = [ [v4[0],v4[1],v4[4]], [v4[0],v4[3],v4[4]],
 [v4[2],v4[1],v4[4]], [v4[2],v4[3],v4[4]], [v4[0],v4[1],v4[2],v4[3]]]

ax.add_collection3d(Poly3DCollection(verts4,
 facecolors='white', linewidths=1, edgecolors='k', alpha=.25))








x=tp[0]/tp[3]
y=tp[1]/tp[3]
z=tp[2]/tp[3]
ax.scatter3D(tp[0]/tp[3], tp[1]/tp[3], tp[2]/tp[3],c=tp[2]/tp[3], marker='o')

ax.set_xlim(-10, 10)
ax.set_ylim(-10,10)
ax.set_zlim(-10,10)


plt.show()





R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cmatrix_left,dis_coeff_left,cmatrix_right,dis_coeff_right,img2.shape[:2],R,T)
np.savetxt('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/R1.txt',R1)
np.savetxt('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/R2.txt',R2)
np.savetxt('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/Q.txt',Q)


(height,width)=img1.shape[:2]
print(height,width)

mapx1, mapy1 = cv2.initUndistortRectifyMap(cmatrix_left,dis_coeff_left,R1,cmatrix_left,(width,height),cv2.CV_32FC1)

mapx2, mapy2 = cv2.initUndistortRectifyMap(cmatrix_right,dis_coeff_right,R2,cmatrix_right,(width,height),cv2.CV_32FC1)

mapx11, mapy11 = cv2.initUndistortRectifyMap(cmatrix_left,dis_coeff_left,None,cmatrix_left,(width,height),cv2.CV_32FC1)

mapx22, mapy22= cv2.initUndistortRectifyMap(cmatrix_right,dis_coeff_right,None,cmatrix_right,(width,height),cv2.CV_32FC1)




np.savetxt('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/R1.txt',R1)
np.savetxt('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/R2.txt',R2)
img_rect1 = cv2.remap(img1, mapx1, mapy1, cv2.INTER_LINEAR)

img_rect2 = cv2.remap(img2, mapx2, mapy2, cv2.INTER_LINEAR)

img_undis_unrect1 = cv2.remap(img1, mapx11, mapy11, cv2.INTER_LINEAR)

img_undis_unrect2 = cv2.remap(img2, mapx22, mapy22, cv2.INTER_LINEAR)

cv2.imwrite('undis_unrect1.png',img_undis_unrect1)
cv2.imwrite('undis_unrect2.png', img_undis_unrect2)

cv2.imshow("imgl",img_undis_unrect1)
cv2.imshow("imgr",img_undis_unrect2)




total_size = (max(img_rect1.shape[0], img_rect2.shape[0]),
                  img_rect1.shape[1] + img_rect2.shape[1], 3)
img = np.zeros(total_size, dtype=np.uint8)
img[:img_rect1.shape[0], :img_rect1.shape[1]] = img_rect1
img[:img_rect2.shape[0], img_rect1.shape[1]:] = img_rect2



cv2.imshow('imgRectified', img)
cv2.imshow("rectified",img)


cv2.waitKey()
