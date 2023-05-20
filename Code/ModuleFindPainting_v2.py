import cv2
import math
import numpy as np
from sympy import Point, Line


def ReplaceColorWithWhite(image, lower_color, upper_color):
    # Converteer de afbeelding naar het HSV-kleursysteem
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #hsv_image = cv2.resize(hsv_image, (int(hsv_image.shape[1] * 5 / 100), int(hsv_image.shape[0] * 5 / 100)), cv2.INTER_AREA)
    
    # Definieer het bereik van kleuren om te vervangen
    lower_bound = np.array(lower_color, dtype=np.uint8)
    upper_bound = np.array(upper_color, dtype=np.uint8)
    
    # Creëer een masker op basis van het kleurbereik
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    
    # Vervang de pixels in het masker door wit
    replaced_image = image.copy()
    replaced_image[mask != 0] = (255, 255, 255)  # Witte kleur
    
    return replaced_image

def CalculateDistance(point1, point2):
    # Bereken de afstand tussen twee punten met de afstandformule
    x_diff = point2[0] - point1[0]
    y_diff = point2[1] - point1[1]
    distance = np.sqrt(x_diff ** 2 + y_diff ** 2)
    return distance

def CheckContourRatio(points, threshold):
    # Bepaal de lengtes van de contourlijnen
    line1 = CalculateDistance(points[0, 0], points[1, 0])
    line2 = CalculateDistance(points[1, 0], points[2, 0])
    line3 = CalculateDistance(points[2, 0], points[3, 0])
    line4 = CalculateDistance(points[3, 0], points[0, 0])
    
    # Bepaal de verhouding tussen de lengtes
    ratio = max(line1, line2, line3, line4) / min(line1, line2, line3, line4)

    # Controleer of de verhouding kleiner is dan de drempelwaarde
    if ratio > threshold:
        return False
    else:
        return True

def CalculateAngle(point1, point2, point3):
    # Bereken de vectoren tussen de punten
    vector1 = (point1[0] - point2[0], point1[1] - point2[1])
    vector2 = (point3[0] - point2[0], point3[1] - point2[1])

    # Bereken de hoek tussen de vectoren met behulp van het inproduct en de normen
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    norm_vector1 = np.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    norm_vector2 = np.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    # Bereken de cosinus van de hoek
    cosine_angle = dot_product / (norm_vector1 * norm_vector2)

    # Bereken de hoek in radialen en converteer naar graden
    angle_rad = math.acos(cosine_angle)
    angle_deg = math.degrees(angle_rad)

    return angle_deg

def CheckCornerAngels(points, lowerThreshold, upperThreshold):
    # Get points
    point1 = points[0, 0]
    point2 = points[1, 0]
    point3 = points[2, 0]
    point4 = points[3, 0]

    # Calculate corners
    corners = []
    corners.append(CalculateAngle(point4, point1, point2))
    corners.append(CalculateAngle(point1, point2, point3))
    corners.append(CalculateAngle(point2, point3, point4))
    corners.append(CalculateAngle(point3, point4, point1))

    # Check if all corners lay between upper and lower threshold
    if max(corners) > upperThreshold or min(corners) < lowerThreshold:
        return False
    else: 
        return True

def CheckParallelogram(points, diffThreshold):
    # Get points
    point1 = points[0, 0]
    point2 = points[1, 0]
    point3 = points[2, 0]
    point4 = points[3, 0]

    # Calculate corners
    corners = []
    corners.append(CalculateAngle(point4, point1, point2))
    corners.append(CalculateAngle(point1, point2, point3))
    corners.append(CalculateAngle(point2, point3, point4))
    corners.append(CalculateAngle(point3, point4, point1))

    # Check if all corners represent parallellogram
    corners = sorted(corners)
    if abs(corners[0] - corners[1]) < diffThreshold and abs(corners[2] - corners[3]) < diffThreshold:
       return True
    else:
       return False

def FiveToFourCorners(points, img):
  # Get corners
  corners = []
  for i in range(5):
      corners.append(points[i, 0])

  # Get points that form shortest line
  shortestDist = np.inf
  point1 = 0
  point2 = 0
  for i in range(len(corners)):
      for j in range(i, len(corners)):
        dist = CalculateDistance(corners[i], corners[j])
        if dist > 0.0 and dist < shortestDist:
          shortestDist = dist
          point1 = i
          point2 = j
  img = cv2.circle(img, corners[point1], 7, [0, 0, 255], 5)
  img = cv2.circle(img, corners[point2], 7, [255, 0, 0], 5)

  # Get next point in contour
  shortestDist1 = np.inf
  point1_next = 0
  for i in range(len(corners)):
    dist = CalculateDistance(corners[point1], corners[i])
    if dist > 0.0 and dist < shortestDist1 and i != point1 and i != point2:
      shortestDist1 = dist
      point1_next = i
  
  shortestDist2 = np.inf
  point2_next = 0
  for i in range(len(corners)):
    dist = CalculateDistance(corners[point2], corners[i])
    if dist > 0.0 and dist < shortestDist2 and i != point1 and i != point2:
      shortestDist2 = dist
      point2_next = i
  
  if point1_next == point2_next:
    if shortestDist1 < shortestDist2:
          shortestDist2 = np.inf
          point2_next = 0
          for i in range(len(corners)):
            dist = CalculateDistance(corners[point2], corners[i])
            if dist > 0.0 and dist < shortestDist2 and i != point1 and i != point2 and i != point1_next:
              shortestDist2 = dist
              point2_next = i
    else:
      shortestDist1 = np.inf
      point1_next = 0
      for i in range(len(corners)):
        dist = CalculateDistance(corners[point1], corners[i])
        if dist > 0.0 and dist < shortestDist1 and i != point1 and i != point2 and i != point2_next:
          shortestDist1 = dist
          point1_next = i
    
  img = cv2.circle(img, corners[point1_next], 7, [0, 0, 255], 5)
  img = cv2.circle(img, corners[point2_next], 7, [255, 0, 0], 5)

  # Line 1
  p1 = Point(corners[point1])
  p2 = Point(corners[point1_next])
  l1 = Line(p1, p2)

  # Line 2
  p3 = Point(corners[point2])
  p4 = Point(corners[point2_next])
  l2 = Line(p3, p4)

  # Find intersection
  intersection = l1.intersection(l2)
  newCorner = [int(intersection[0].x), int(intersection[0].y)]
  img = cv2.circle(img, (int(intersection[0].x), int(intersection[0].y)), 7, [0, 255, 255], 5)

  # Change corners of contour
  for i in range(len(corners)):
    if i != point1 and i != point1_next and i != point2 and i != point2_next:
      lastCorner = corners[i]
  newCorners = np.array([[newCorner],[ corners[point1_next]], [lastCorner], [corners[point2_next]]])
  
  return newCorners

def CheckPositionOfExtraxt(points, imgShape, border):
  result = True
  for point in points:
    if point[0, 0] < border or point[0, 0] > imgShape[1] - border or point[0, 1] < border or point[0, 1] > imgShape[0] - border:
      result = False
  return result



def FindPainting(img):
  # Covert to gray
  img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

  # Bilateral filter
  size = 7
  kernel = np.ones((size, size), np.float32) / (size * size)
  img_bil = cv2.filter2D(img_gray, -1, kernel)
  
  # Otsu thresholding
  ret, img_otsu = cv2.threshold(img_bil, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  # Dilate
  size = 13
  kernel = np.ones((size, size), np.float32) / (size * size)
  img_dilate = cv2.dilate(img_otsu, kernel)

  # Canny
  img_canny = cv2.Canny(img_dilate, 5, 5)

  # Dilate
  size = 30
  kernel = np.ones((size, size), np.float32) / (size * size)
  img_dil = cv2.dilate(img_canny, kernel, iterations=1)

  # Find contours
  contours, _ = cv2.findContours(img_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Filter contours
  contours_list = []
  area_threshold = 0.10
  contours = sorted(contours, key=cv2.contourArea, reverse=True)
  contour_max = max(contours, key=cv2.contourArea)
  contour_max_area = cv2.contourArea(contour_max)
  for contour in contours:
    # Filteren op oppervlakte grootte
    area = cv2.contourArea(contour)
    if area < contour_max_area * area_threshold:
      continue
    
    # Contour toevoegen aan lijst
    contours_list.append(contour)

  # Enkel buitenste punten nemen uit contour
  hull_list = []
  for contour in contours_list:
    hull = cv2.convexHull(contour)
    hull_list.append(hull)

  # Fit quadrilateral
  threshold_ratio = 3
  threshold_lowerAngle = 80
  threshold_upperAngle = 100
  threshold_diffAngle = 10

  quadrilateral_list = []
  for hull in hull_list:
    approx = cv2.approxPolyDP(hull, 0.015 * cv2.arcLength(hull, True), True)
    if len(approx) == 4:
      if CheckContourRatio(approx, threshold_ratio) and CheckParallelogram(approx, threshold_diffAngle):
        quadrilateral_list.append(approx)
    if len(approx) == 5:
      newContour = FiveToFourCorners(approx, img)
      if newContour is not None: 
        if CheckContourRatio(newContour, threshold_ratio) and CheckParallelogram(newContour, threshold_diffAngle) and CheckCornerAngels(newContour, threshold_lowerAngle, threshold_upperAngle):
          quadrilateral_list.append(newContour)

  # Filter out bad extraxts and raw quadrilateral
  threshold_border = 5
  imgContour = img.copy()
  goodContours = []
  for quadrilateral in quadrilateral_list:
    if CheckPositionOfExtraxt(quadrilateral, img.shape, threshold_border):
      cv2.drawContours(imgContour, [quadrilateral], -1, (0, 255, 0), 5)
      goodContours.append(quadrilateral)
    else:
      cv2.drawContours(imgContour, [quadrilateral], -1, (0, 0, 255), 5)

  # Extract painting
  extraxtList = []
  for goodContour in goodContours:
    x = []
    y = []
    for corner in goodContour:
      x.append(corner[0, 0])
      y.append(corner[0, 1])
    extraxt = img.copy()
    extraxtList.append(extraxt[min(y):max(y), min(x):max(x)])

  return imgContour, extraxtList

