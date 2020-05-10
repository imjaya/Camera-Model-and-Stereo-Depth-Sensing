# Camera-Model-and-Stereo-Depth-Sensing

# Task 1:
Pinhole Camera Model and Calibration:
• In this task we have used the inbuilt python library ‘glob’ to import all the images in the task 1 folder into a single object with a variable name and then iterate through it to read individual images using the OpenCV function ‘imread()’.
• After using the ‘findChessboardCorners()’ function to determine all the corners and filling the 6 by 9 grid we use the ‘drawChessboardCorners()’ function in OpenCV to draw the corners on the image shown in the results.
• We obtained the camera intrinsic parameters and the distortion vector of the left and right cameras using the ‘calibrateCamera()’ function.
• These intrinsic parameters and the distortion vector have been stored as individual CSV files using ‘np.savetxt()’ numpy function.

# Task 2:
Stereo Calibration and Rectification:
• We first imported the left and right camera’s intrinsic matrices and the distortion vectors.
• We then again find the 3D – 2D point correspondences using the ‘findChessboardCorners()’ function.
• Using these object points(3D points) and left and right image points(2D points) we calibrate the stereo cameras using the function ‘stereoCalibrate()’ to get the R, T, E and F matrices.
• Now to plot the results we first undistort the points using ‘undistortPoints()’ function, apply the scale transformation to the T vector and then plot using ‘triangulatePoints()’ function.
• Now in order to apply the R and T transforms to the cameras we use the function ‘stereoRectify()’. This gives the rotation matrices for the left and right camera such that there is only the translation between the two cameras.

# Task 3:
Sparse Depth Triangulation:
• We first load the camera intrinsic matrices and distortion vectors.
• We then detect ORB feature points and record the response value for every feature point and take the argmax of a window of 6 pixels and draw that feature point using the function ‘drawKeypoints()’.
• We now draw the matchings between the local maxima feature points from two different views by using the function ‘drawMatches()’.
• We use the ‘triangulatePoints()’ function to plot the 3D points using the matches.
• We now plot the center of the left camera at the origin and using the R and T obtained from stereo calibration we plot the center of the right camera along with the 3D points.

# Task 4:
Dense Depth Triangulation:
• We first load the camera intrinsic matrices and distortion vectors.
• We first undistort the image and then use the ‘StereoSGBM’ class in OpenCV we get a disparity map for every pixel i.e. we get the |ul - ur| value for every pixel.
• Now using this disparity map we generate a depth map using the function ‘reprojectImageTo3D()’ function with a baseline length of 62mm.
