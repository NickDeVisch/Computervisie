import os
import cv2
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.seterr(invalid ='ignore')
np.seterr(over ='ignore')
warnings.filterwarnings("ignore", category=UserWarning)

from ModuleGetData import GetDataFromDrive
from ModuleFindPainting_v2 import FindPainting, CheckCornersRelativeToPrevious
from ModuleMatcher import Matching
from ModuleFloorPlan import Floorplan
from ModuleDisplayScreen import DisplayScreen, ResizeImage
from test import create_chart_from_dataframe

# Default url
url = 'D:\\School\\UGent\\AUT 5\\Computervisie\\Computervisie'

if __name__ == '__main__':
    # Load video
    videoUrl =  url + '\\Videos\\GoPro\\MSK_17.mp4'
    video = cv2.VideoCapture(videoUrl)
    
    # Init objects
    getDataFromDrive = GetDataFromDrive(url)
    matching = Matching(getDataFromDrive.keypoints, getDataFromDrive.descriptors, getDataFromDrive.df, url)
    floorPlan = Floorplan(url)
    displayScreen = DisplayScreen(videoUrl)

    # Init variables
    previousCorners = None

    # Iterate over frames
    for i in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
        # Get frame from video and undistort it
        ret, frame = video.read()
        frame = displayScreen.UndistortFrame(frame)

        # Detect paintings in frame
        frame, extraxtList, corners = FindPainting(frame, matching.roomSequence)

        # Check if any extraxts where found
        if len(extraxtList) != 0:
            # Match detected paintings from frame
            matches = pd.DataFrame()
            for extraxt in extraxtList:
                result = matching.MatchPainting(extraxt)
                matches = pd.concat([matches, result[:1]])
            
            # Take best match
            bestMatch = matches.sort_values(by=['total'], ascending=False)
            
            # Check if match is good enough
            if bestMatch['total'].values[0] > 0.35:
                # Add room to roomSequence
                matching.AppendRoom(bestMatch['naam'].values[0].split('__')[0]) 

                # Get matching painting from database and print name in it
                matchPainting = ResizeImage(cv2.imread(url + '\\Database\\' + bestMatch['naam'].values[0]))
                matchPainting = cv2.putText(matchPainting, bestMatch['naam'].values[0], (5, 25), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA, False)

                # Update floorplan
                floorplan = floorPlan.DrawPath(matching.roomSequence)

                # Generate windows
                cv2.imshow('Best match', matchPainting)
                cv2.imshow('Floorplan', floorplan)

        # Save last corners
        previousCorners = corners

        # Show frame with contours
        cv2.imshow('Video', ResizeImage(frame))
        cv2.waitKey(1)

    # Destroy all windows when video ends
    cv2.destroyAllWindows()   
