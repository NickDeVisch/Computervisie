import os
import cv2
import time
import collections
import numpy as np
import pandas as pd

from ModuleGetData import GetDataFromDrive
from ModuleFindPainting_v2 import FindPainting, ReplaceColorWithWhite
from ModuleMatcher import Matching
from ModuleFloorPlan import Floorplan
from ModuleDisplayScreen import ResizeImage

if __name__ == '__main__':
    url = 'D:\\School\\UGent\\AUT 5\\Computervisie\\Computervisie'

    getDataFromDrive = GetDataFromDrive(url)
    matching = Matching(getDataFromDrive.keypoints, getDataFromDrive.descriptors, getDataFromDrive.df, url)
    floorPlan = Floorplan(url)

    cameraMatrix = np.array([[582.02639453, 0., 647.52365408], [0., 586.04899393, 339.20774435],[0., 0., 1.]])
    distCoeffs = np.array([[-2.42003542e-01,  7.01396093e-02, -8.30073220e-04, 9.71570940e-05, -1.02586096e-02]])
    
    cameraMatrix = np.array([[722.31231717, 0., 648.09282601], [0., 727.65628288, 323.11790409], [0., 0., 1.]])
    distCoeffs = np.array([[-0.26972165, 0.11073541, 0.00049764, -0.00060387, -0.02801339]])

    # Load video
    videoUrl =  url + '\\Videos\\GoPro\\MSK_16.mp4'
    video = cv2.VideoCapture(videoUrl)

    goodMatch = False
    for i in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
        # Get frame from video abd undistort it
        ret, frame = video.read()
        frame = cv2.undistort(frame, cameraMatrix, distCoeffs)

        if goodMatch:
            if i%180 == 0: 
                goodMatch = False
        if not goodMatch:
            if i%5 != 0: continue

        #print('Frame', i)
        frame, extraxtList = FindPainting(frame, 'Zaal 5')
        if goodMatch == False and False:
            goodMatches = pd.DataFrame()
            for extraxt in extraxtList:
                matchResult = matching.MatchPainting(extraxt)
                flannAmount_1 = matchResult[:1]['flannAmount'].values[0]
                flannAmount_2 = matchResult[1:2]['flannAmount'].values[0]
                if (flannAmount_1 > 2.5 * flannAmount_2 and flannAmount_1 > 75) or flannAmount_1 > 150:
                    goodMatch = True
                    goodMatches = pd.concat([goodMatches, matchResult[:1]])
            
            if goodMatch:
                # Get best match and add to room sequence
                matchResult = goodMatches[goodMatches['flannAmount'] == goodMatches['flannAmount'].max()]
                #tempRooms=collections.deque(["dummy1", "dummy2", "dummy3", "dummy4", "dummy5"])             #eerste werkelijke zaal zal naar buiten gebracht worden
                #tempRooms.appendleft(matchResult['naam'].values[0].split('__')[0])
                #tempRooms.pop
                #freq=collections.Counter(tempRooms)
                #mostFreq,_=freq.most_common(1)[0]
                #matching.AppendRoom(mostFreq) #NIET ZEKER VAN DEZE IMPLEMENTATIE
                matching.AppendRoom(matchResult['naam'].values[0].split('__')[0]) 

                # Get matching painting from database and print name in it
                matchPainting = cv2.imread(url + '\\Database\\' + matchResult['naam'].values[0])
                matchPainting = cv2.putText(matchPainting, matchResult['naam'].values[0], (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10, cv2.LINE_AA, False)
                matchPainting = ResizeImage(matchPainting)

                # Update floorplan
                floorplan = floorPlan.DrawPath(matching.roomSequence)

                # Generate windows
                cv2.imshow('Best match', matchPainting)
                cv2.imshow('Extract', ResizeImage(extraxt))
                cv2.imshow('Floorplan', floorplan)

        cv2.imshow('Video', ResizeImage(frame))
        cv2.waitKey(1)
    cv2.destroyAllWindows()   
