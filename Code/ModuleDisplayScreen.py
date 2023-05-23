import os
import cv2
import numpy as np

def ResizeImage(img):
    return cv2.resize(img, [int(img.shape[1] / img.shape[0] * 400), 400], cv2.INTER_AREA)


class DisplayScreen:
    cameraMatrix_M = np.array([[722.31231717, 0., 648.09282601], [0., 727.65628288, 323.11790409], [0., 0., 1.]])
    distCoeffs_M = np.array([[-0.26972165, 0.11073541, 0.00049764, -0.00060387, -0.02801339]])

    cameraMatrix_W = np.array([[582.02639453, 0., 647.52365408], [0., 586.04899393, 339.20774435],[0., 0., 1.]])
    distCoeffs_W = np.array([[-2.42003542e-01,  7.01396093e-02, -8.30073220e-04, 9.71570940e-05, -1.02586096e-02]])
    
    cameraMatrix = None
    distCoeffs = None

    def __init__(self, videoUrl):   
        videoNumber = int(videoUrl.split('\\')[-1].split('.')[0].split('_')[1])
        if videoNumber == 12 or videoNumber == 18 or videoNumber == 19:
            self.cameraMatrix = self.cameraMatrix_M
            self.distCoeffs = self.distCoeffs_M
        else:
            self.cameraMatrix = self.cameraMatrix_W
            self.distCoeffs = self.distCoeffs_W

    def UndistortFrame(self, frame):
        if self.cameraMatrix is None and self.distCoeffs is None:
            return frame
        else:
            frame = cv2.undistort(frame, self.cameraMatrix, self.distCoeffs)
            return frame
