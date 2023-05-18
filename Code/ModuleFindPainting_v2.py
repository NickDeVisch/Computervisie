import cv2
import numpy as np

def FindCornersPainting(img):
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
  quadrilateral_list = []
  for hull in hull_list:
    approx = cv2.approxPolyDP(hull, 0.015 * cv2.arcLength(hull, True), True)
    if len(approx) == 4:
      quadrilateral_list.append(approx)
    print(len(quadrilateral_list))

  # Draw quadrilateral
  img_copy = img.copy()
  cv2.drawContours(img_copy, quadrilateral_list, -1, (0, 255, 0), 5)

  return [img_copy]


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

