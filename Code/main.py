import os
import cv2
import time
import numpy as np
import pandas as pd

from ModuleGetData import GetDataFromDrive
from ModuleFindPainting import FindPainting
from ModuleMatcher import Matching
from ModuleFloorPlan import Floorplan
from ModuleDisplayScreen import ResizeImage

url = 'D:\\School\\UGent\\AUT 5\\Computervisie\\Computervisie'

getDataFromDrive = GetDataFromDrive(url)
findPainting = FindPainting()
matching = Matching(getDataFromDrive.keypoints, getDataFromDrive.descriptors, getDataFromDrive.df, url)
floorPlan = Floorplan(url)

cameraMatrix = np.array([[582.02639453, 0., 647.52365408], [0., 586.04899393, 339.20774435],[0., 0., 1.]])
distCoeffs = np.array([[-2.42003542e-01,  7.01396093e-02, -8.30073220e-04, 9.71570940e-05, -1.02586096e-02]])

# Load video
videoUrl =  url + '\\Videos\\GoPro\\MSK_16.mp4'
video = cv2.VideoCapture(videoUrl)

goodMatch = False
for i in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
    # Get frame from video, undistort and resize it
    ret, frame = video.read()
    frame = cv2.undistort(frame, cameraMatrix, distCoeffs)
    frame = cv2.resize(frame, (int(frame.shape[1] * 50 / 100), int(frame.shape[0] * 50 / 100)), cv2.INTER_AREA)

    if goodMatch:
        if i%360 == 0: 
            goodMatch = False
    if not goodMatch:
        if i%5 != 0: continue

    if goodMatch == False:
        print('Frame', i, 'Finding painting')
        extraxtList = findPainting.FindPainting(frame)
        if len(extraxtList): print('Matching, Paintings found:', len(extraxtList))
        goodMatches = pd.DataFrame()
        for extraxt in extraxtList:
            matchResult = matching.MatchPainting(extraxt)
            flannAmount_1 = matchResult[:1]['flannAmount'].values[0]
            flannAmount_2 = matchResult[1:2]['flannAmount'].values[0]
            print(flannAmount_1, flannAmount_2)
            if (flannAmount_1 > 2.5 * flannAmount_2 and flannAmount_1 > 150) or flannAmount_1 > 200:
                goodMatch = True
                goodMatches = pd.concat([goodMatches, matchResult[:1]])
        
        if goodMatch:
            matchResult = goodMatches[goodMatches['flannAmount'] == goodMatches['flannAmount'].max()]
            matching.AppendRoom(matchResult['naam'].values[0].split('__')[0])

            matchPainting = ResizeImage(cv2.imread(url + '\\Database\\' + matchResult['naam'].values[0]))
            floorplan = floorPlan.DrawPath(matching.roomSequence)

            cv2.imshow('Best match', matchPainting)
            cv2.imshow('Extract', extraxt)
            cv2.imshow('Floorplan', floorplan)

    cv2.imshow('Video', ResizeImage(frame))
    cv2.waitKey(1)
cv2.destroyAllWindows()   
