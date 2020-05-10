import cv2
import numpy as np
import glob

rows = 6
cols = 9
###############################################################################
                               # LEFT
###############################################################################
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)


objpoints_l = [] # 3d point in real world space
imgpoints_l = [] # 2d points in image plane.

images = glob.glob('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/images/task_1/left*.png')

for img_name in images:
    img_color = cv2.imread(img_name)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(img, (rows, cols), None)
    objpoints_l.append(objp)
    imgpoints_l.append(corners)
    if ret == True:
        corners2 = cv2.cornerSubPix(img, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        cv2.drawChessboardCorners(img_color, (rows, cols), corners2, ret)
        cv2.imshow('img', img_color)
        cv2.waitKey(1000)
cv2.destroyAllWindows()

   #Calibration

ret, K, dist, _, _ = cv2.calibrateCamera(objpoints_l, imgpoints_l, img.shape[::-1], None, None)

if ret == False:
    print("Camera images not present for camera number")

# Undistortion
i = 0
# for i in range(1):
img = cv2.imread('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/images/task_1/left_2.png')

h, w = img.shape[:2]
print(dist)
newK_left, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 0)
print(roi)

mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, newK_left, (w,h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
if dst[0].size > 0:
    cv2.imwrite('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/output/task_1/left_2_undistorted.png', dst)
    undis_img = cv2.imread('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/output/task_1/left_2_undistorted.png')
    cv2.imshow('undistorted image', undis_img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
else:
    print('image zero')

cv2.destroyAllWindows()

np.savetxt('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/new_K_matrix_left', newK_left)
np.savetxt('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/K_matrix_left', K)
np.savetxt('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/distortion_coeffs_left', dist)

###############################################################################
                        # RIGHT SIDE
###############################################################################



rows = 9
cols = 6

objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)


objpoints_r = [] # 3d point in real world space
imgpoints_r = [] # 2d points in image plane.

images = glob.glob('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/images/task_1/right*.png')

for img_name in images:
    img_color = cv2.imread(img_name)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(img, (rows, cols), None)
    objpoints_r.append(objp)
    imgpoints_r.append(corners)
    if ret == True:
        corners2 = cv2.cornerSubPix(img, corners, (5,5), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        cv2.drawChessboardCorners(img_color, (rows, cols), corners2, ret)
        cv2.imshow('img', img_color)
        cv2.waitKey(1000)
cv2.destroyAllWindows()

   #Calibration

ret, K_right, dist_right, _, _ = cv2.calibrateCamera(objpoints_r, imgpoints_r, img.shape[::-1], None, None)

if ret == False:
    print("Camera images not present for camera number")

# Undistortion
# i = 0
# for i in range(1):
img = cv2.imread('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/images/task_1/right_2.png')

h, w = img.shape[:2]
print(dist)
newK_right, roi = cv2.getOptimalNewCameraMatrix(K_right, dist_right, (w,h), 0)
print(roi)

mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, newK_right, (w,h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
if dst[0].size > 0:
    cv2.imwrite('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/output/task_1/right_2_undistorted.png', dst)
    undis_img = cv2.imread('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/output/task_1/right_2_undistorted.png')
    cv2.imshow('undistorted image', undis_img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
else:
    print('image zero')

cv2.destroyAllWindows()

np.savetxt('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/new_K_matrix_right', newK_right)
np.savetxt('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/K_matrix_right', K_right)
np.savetxt('D:/SPRING 2020/CSE 598 Perception in Robotics/Project/Project 2a/project_2a/parameters/distortion_coeffs_right', dist_right)
