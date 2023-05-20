import cv2
import numpy as np

f_face = [[], [], []]
r_face = [[], [], []]
b_face = [[], [], []]
l_face = [[], [], []]
u_face = [[], [], []]
d_face = [[], [], []]
cube_colors = {"f": f_face, "r": r_face, "b": b_face, "l": l_face, "u": u_face, "d": d_face}
cube_done = False;

green_lower = (40, 50, 50)  # lower bound for green color in HSV
green_upper = (80, 255, 255)  # upper bound for green color in HSV
red_lower = (60, 100, 100)
red_upper = (70, 255, 255)
blue_lower = (90, 50, 50)
blue_upper = (150, 255, 255)
yellow_lower = (60, 100, 100)
yellow_upper = (40, 255, 255)
orange_lower = (10, 100, 90)
orange_upper = (50, 255, 255)
white_lower = (0, 0, 180)
white_upper = (180, 30, 255)

min_square_area = 2000
max_square_area = 40000

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
    
    # Apply thresholding to enhance contour detection
    _, threshold = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours in the thresholded mask
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and refine contours
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if max_square_area > area > min_square_area:  # Filter contours based on minimum area
            # Find the bounding rectangle and calculate its aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            # Define the acceptable range for aspect ratio of squares (e.g., 0.9 to 1.1)
            aspect_ratio_range = (0.9, 1.1)
            
            if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
                filtered_contours.append((x, y, w, h))
    
    # Check for overlapping square regions
    overlapping_squares = []
    for i in range(len(filtered_contours)):
        for j in range(i+1, len(filtered_contours)):
            xi, yi, wi, hi = filtered_contours[i]
            xj, yj, wj, hj = filtered_contours[j]
            
            # Check if the squares have overlapping regions
            if xi < xj+wi and xi+wi > xj and yi < yj+hj and yi+hi > yj:
                overlapping_squares.append((xi, yi, wi, hi))
                overlapping_squares.append((xj, yj, wj, hj))
                
    # Draw rectangles for overlapping square regions
    for square in overlapping_squares:
        x, y, w, h = square
        cv2.rectangle(frame, (x, y), (x+w, y+h), (55, 100, 80), 2)
        
    # Display the resulting frame
    cv2.imshow('Cube Solver', frame)
      
    # the 'q' button is set as the quitting button 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
cam.release()
# Destroy all the windows
cv2.destroyAllWindows()