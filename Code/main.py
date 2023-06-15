# Import libraries
import os
import cv2
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#TEST

# Change settings
np.seterr(invalid ='ignore')
np.seterr(over ='ignore')
warnings.filterwarnings("ignore", category=UserWarning)

# Import modules
from ModuleGetData import GetDataFromDrive
from ModuleFindPainting_v2 import FindPainting, CheckCornersRelativeToPrevious
from ModuleMatcher import Matching
from ModuleFloorPlan import Floorplan
from ModuleDisplayScreen import DisplayScreen, ResizeImage, CheckSharpnessOfImage

# Default url
url = 'D:\\School\\UGent\\AUT 5\\Computervisie\\Computervisie'
#url = 'D:\\1_School\\Ugent\\Masterjaar\\2_Computervisie\\Project'  #LOUIS

# Main
if __name__ == '__main__':
    # Load video
    videoUrl =  url + '\\Videos\\GoPro\\MSK_16.mp4'
    #videoUrl =  url + '\\Videos\\Smartphone\\MSK_02.mp4'   #LOUIS
    video = cv2.VideoCapture(videoUrl)
    
    # Init objects
    getDataFromDrive = GetDataFromDrive(url)
    matching = Matching(getDataFromDrive.keypoints, getDataFromDrive.descriptors, getDataFromDrive.df, url)
    floorPlan = Floorplan(url)
    displayScreen = DisplayScreen(videoUrl)

    # Init variables
    previousCorners = None
    goodMatch = False

    # Iterate over frames
    for i in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
        # Get frame from video and undistort it
        ret, frame = video.read()
        frame = displayScreen.UndistortFrame(frame)

        # Detect paintings in frame
        frame, extraxtList, corners = FindPainting(frame, matching.lastMatches)

        # Skip matching for frames
        if goodMatch:
            if i%180 == 0:
                goodMatch = False

        # Check if any extraxts where found
        if len(extraxtList) != 0 and not goodMatch and i%5 == 0:
            # Match detected paintings from frame
            matches = pd.DataFrame()
            for extraxt in extraxtList:
                # Check extraxt on sharpness
                if extraxt.size == 0: continue
                if not CheckSharpnessOfImage(extraxt, 50.0): continue
                
                # Match extraxt
                result = matching.MatchPainting(extraxt)
                matches = pd.concat([matches, result[:1]])
            
            # Check if match has been found
            if len(matches) > 0:
                # Take best match
                bestMatch = matches.sort_values(by=['total'], ascending=False).reset_index()
                print(bestMatch)

                # Check if match is good enough
                if bestMatch['total'].values[0] > 0.45:
                    # Add room to roomSequence if it is a good room
                    goodMatch = matching.AppendRoom(bestMatch['naam'].values[0].split('__')[0]) 

                    # Get matching painting from database and print name in it
                    matchPainting = ResizeImage(cv2.imread(url + '\\Database\\' + bestMatch['naam'].values[0]))
                    matchPainting = cv2.putText(matchPainting, 'Score: ' + str(round(bestMatch['total'].values[0], 5)), (5, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA, False)
                    if goodMatch: matchPainting = cv2.putText(matchPainting, bestMatch['naam'].values[0], (5, 25), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA, False)
                    else: matchPainting = cv2.putText(matchPainting, bestMatch['naam'].values[0], (5, 25), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA, False)

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
    print('Video done!')
    cv2.waitKey(0)
    cv2.destroyAllWindows()   
