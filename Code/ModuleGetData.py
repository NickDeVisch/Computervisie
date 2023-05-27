import os
import cv2
import pickle
import numpy as np
import pandas as pd

from ModuleDisplayScreen import ResizeImage

from ModuleFeatureVector import ColorHist
from ModuleFeatureVector import ComRed
from ModuleFeatureVector import ColorPatches

class GetDataFromDrive:
  debug = False
  url = ''
  
  df = pd.DataFrame()
  keypoints = {}
  descriptors = {}


  def __init__(self, url, debug=False):
    if self.debug: print('Init starting...')
    self.url = url
    self.debug = debug

    # DataFrame
    if os.path.isfile(self.url + '\\df.pkl'):
      self.__OpenDF()
    else:
      self.__CreateDF()

    # Keypoints and descriptors
    if os.path.isfile(self.url + '\\keypoints.pkl') and os.path.isfile(self.url + '\\descriptors.pkl'):
      self.__OpenKeyAndDesc()
    else:
      self.__CreateKeyAndDesc()

    if self.debug: print('Init done!')


  def __CreateDF(self):
    if self.debug: print('Creating DataFrame...')
    namen = []
    zalen = []
    colorHist = []
    cX = []
    cY = []
    colorPatch = []

    for i, name in enumerate(os.listdir(self.url + '\\Database')):
      if self.debug: print(i)

      img = cv2.imread(self.url + '\\Database' + '\\' + name)
      img = ResizeImage(img, 1800)

      namen.append(name)
      
      parts = name.split("_")
      zalen.append(parts[0].lower() + '_' + parts[1])

      colorHist.append(ColorHist(img))

      temp_cX, temp_cY = ComRed(img)
      cX.append(temp_cX)
      cY.append(temp_cY)

      colorPatch.append(ColorPatches(img))

    self.df['naam'] = namen
    self.df['zaal'] = zalen
    self.df['hist'] = colorHist
    self.df['cX'] = cX
    self.df['cY'] = cY
    self.df['patch'] = colorPatch

    with open(self.url + '\\df.pkl', 'wb') as fid:
      pickle.dump(self.df, fid)


  def __OpenDF(self):
    if self.debug: print('Opening DataFrame...')
    with open(self.url + '\\df.pkl', 'rb') as fid:
      self.df = pickle.load(fid)


  def __CreateKeyAndDesc(self):
    if self.debug: print('Creating files...')

    for i, painting in enumerate(os.listdir(self.url + '\\Database')):
      if self.debug: print(i)

      img = cv2.imread(self.url + '\\Database' + "\\"  + painting)
      img_copy = cv2.resize(img, (int(img.shape[1]/4), int(img.shape[0]/4)))
      img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
      img_copy = cv2.GaussianBlur(img_copy, (3, 3), 0)

      sift = cv2.SIFT_create(nfeatures=500, nOctaveLayers=6, contrastThreshold=0.04, edgeThreshold=35, sigma=2.2)
      key_point, descr = sift.detectAndCompute(img_copy, None)
      descr = descr.astype(np.float32)
      
      self.keypoints[painting] = [(kp.pt[0], kp.pt[1]) for kp in key_point]
      self.descriptors[painting] = descr

    with open(self.url + '\\keypoints.pkl', 'wb') as fid:
      pickle.dump(self.keypoints, fid)

    with open(self.url + '\\descriptors.pkl', 'wb') as fid:
      pickle.dump(self.descriptors, fid)


  def __OpenKeyAndDesc(self):
    if self.debug: print('Opening files...')
    with open(self.url + '\\keypoints.pkl', 'rb') as fid:
      keypoints = pickle.load(fid)

    for painting, kps in keypoints.items():
      kp_objects = []
      for x, y in kps:
          kp = cv2.KeyPoint(x, y, 1)
          kp_objects.append(kp)
      self.keypoints[painting] = kp_objects

    with open(self.url + '\\descriptors.pkl', 'rb') as fid:
      self.descriptors = pickle.load(fid)
