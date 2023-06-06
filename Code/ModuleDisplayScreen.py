import os
import cv2
import numpy as np

# Resize image to certain size
def ResizeImage(img, size=400):
    return cv2.resize(img, [int(img.shape[1] / img.shape[0] * size), size], cv2.INTER_AREA)

# Check sharpness of image
def CheckSharpnessOfImage(img, threshold, debug=False):
    # Convert image to gray 
    img_copy = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculatie laplacion variance
    laplacian_variatie = cv2.Laplacian(img_copy, cv2.CV_64F).var()
    
    # Print laplacion variance
    if debug: print(laplacian_variatie)
    
    # Return True if variation is bigger than givven threshold
    if laplacian_variatie > threshold:
        return True
    else:
        return False


class DisplayScreen:
    # Parameters camera M
    cameraMatrix_M = np.array([[722.31231717, 0., 648.09282601], [0., 727.65628288, 323.11790409], [0., 0., 1.]])
    distCoeffs_M = np.array([[-0.26972165, 0.11073541, 0.00049764, -0.00060387, -0.02801339]])

    # Parameters camera W
    cameraMatrix_W = np.array([[582.02639453, 0., 647.52365408], [0., 586.04899393, 339.20774435],[0., 0., 1.]])
    distCoeffs_W = np.array([[-2.42003542e-01,  7.01396093e-02, -8.30073220e-04, 9.71570940e-05, -1.02586096e-02]])
    
    cameraMatrix = None
    distCoeffs = None

    def __init__(self, videoUrl):
        # Determine video from name and set correct camera parameters
        # In case of smartphone, the parameters aren't changed
        if videoUrl.split('\\')[-2] == 'GoPro':
            videoNumber = int(videoUrl.split('\\')[-1].split('.')[0].split('_')[1])
            if videoNumber == 12 or videoNumber == 18 or videoNumber == 19:
                self.cameraMatrix = self.cameraMatrix_M
                self.distCoeffs = self.distCoeffs_M
            else:
                self.cameraMatrix = self.cameraMatrix_W
                self.distCoeffs = self.distCoeffs_W

    def UndistortFrame(self, frame):
        # Undistort frame with set parameters
        if self.cameraMatrix is None and self.distCoeffs is None:
            return frame
        else:
            frame = cv2.undistort(frame, self.cameraMatrix, self.distCoeffs)
            return frame
