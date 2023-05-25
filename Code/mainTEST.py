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
from ModuleMatcher import Matching
from ModuleFloorPlan import Floorplan
from ModuleDisplayScreen import DisplayScreen, ResizeImage

# Default url
url = 'D:\\1_School\\Ugent\\Masterjaar\\2_Computervisie\\Project'  #LOUIS  

# Init objects
getDataFromDrive = GetDataFromDrive(url)
matching = Matching(getDataFromDrive.keypoints, getDataFromDrive.descriptors, getDataFromDrive.df, url)

# Init variables
previousCorners = None
goodMatch = False

def diff_second(i,path):
  j=0
  count=0
  input_name=[]
  matching_name=[]
  near_second=[]
  matching_score=[]

  for file in os.listdir(path):
    if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
      count += 1
      if (count==i):
        count=0
        value_input_name=file
        input_name.append(value_input_name)
        print(file)  #image die getest wordt 
        img = cv2.imread(path+"/"+file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img=ResizeImage(img)
        df_test=matching.MatchPainting(img) 
        df_test = df_test.reset_index()
        del df_test['index']
        value_matching_name=df_test.iat[0,1]
        matching_name.append(value_matching_name)
        value_img_score= df_test.iat[0,7]
        matching_score.append(value_img_score)
        value_near_second= df_test.iat[0,7]-df_test.iat[1,7]
        near_second.append(value_near_second)
        print("iteratie: ",j)
        j=j+1
        print(matching_name)
        print(near_second)
        print(matching_score)

  return matching_name,matching_score,near_second  
url = 'D:\\1_School\\Ugent\\Masterjaar\\2_Computervisie\\Project\\Database'
#diff_second(1,url)


