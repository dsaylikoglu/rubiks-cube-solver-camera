import cv2
import numpy as np

f_face = [[], [], []]
r_face = [[], [], []]
b_face = [[], [], []]
l_face = [[], [], []]
u_face = [[], [], []]
d_face = [[], [], []]

green_lower = (40, 50, 50)  # lower bound for green color in HSV
green_upper = (80, 255, 255)  # upper bound for green color in HSV
red_lower = (0, 100, 100)
red_upper = (10, 255, 255)
blue_lower = (90, 50, 50)
blue_upper = (130, 255, 255)
yellow_lower = (20, 100, 100)
yellow_upper = (40, 255, 255)
orange_lower = (0, 100, 100)
orange_upper = (20, 255, 255)
white_lower = (0, 0, 200)
white_upper = (180, 50, 255)

# define a video capture object
cam = cv2.VideoCapture(0)
  
while(True):
    # Capture the video frame by frame
    ret, frame = cam.read()
    
    if not ret:
      break
  
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Masks are color recognizers
    green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)
    red_mask = cv2.inRange(hsv_frame, red_lower, red_upper)
    blue_mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)
    yellow_mask = cv2.inRange(hsv_frame, yellow_lower, yellow_upper)
    orange_mask = cv2.inRange(hsv_frame, orange_lower, orange_upper)
    white_mask = cv2.inRange(hsv_frame, white_lower, white_upper)
    
    # Combine the masks
    combined_mask = cv2.bitwise_or(red_mask, green_mask)
    combined_mask = cv2.bitwise_or(combined_mask, blue_mask)
    combined_mask = cv2.bitwise_or(combined_mask, yellow_mask)
    combined_mask = cv2.bitwise_or(combined_mask, orange_mask)
    combined_mask = cv2.bitwise_or(combined_mask, white_mask)
  
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the total mask (min areas of color)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding rectangles around the detected regions
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (67, 155, 100), 2)
        
    # Display the resulting frame
    cv2.imshow('Camera', frame)
      
    # the 'q' button is set as the quitting button 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
cam.release()
# Destroy all the windows
cv2.destroyAllWindows()