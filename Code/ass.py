import os
import cv2
import numpy as np
import pandas as pd

from ModuleFindPainting_v2 import FindPainting
from ModuleDisplayScreen import ResizeImage


df = pd.read_csv('D:\\School\\UGent\\AUT 5\\Computervisie\\Computervisie\\ass_1.csv')
print(df[df['resultaat'] == True]['hoeken'])







exit(0)

# Code om eerste csv te maken
url = 'D:\\School\\UGent\\AUT 5\\Computervisie\\Computervisie\\TrainData'

zalen = []
namen = []
result = []
hoeken = []

for i, zaal in enumerate(os.listdir(url)):
    print(zaal)
    for j, naam in enumerate(os.listdir(url + '\\' + zaal)):
        img = cv2.imread(url + '\\' + zaal + '\\' + naam)
        img, extraxtList, corners = FindPainting(img, [zaal])
        
        zalen.append(zaal)
        namen.append(naam)
        if corners == []: result.append(False)
        else: result.append(True)
        hoeken.append(corners)

df = pd.DataFrame()
df['zaal'] = zalen
df['naam'] = namen
df['resultaat'] = result
df['hoeken'] = hoeken
print(df)
df.to_csv('D:\\School\\UGent\\AUT 5\\Computervisie\\Computervisie\\ass_1.csv', index = False)