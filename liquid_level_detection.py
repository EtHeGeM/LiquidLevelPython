import cv2
import numpy as np

def calculate_water_level(frame, lower_color, upper_color):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the color in HSV
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Find contours in the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area (assumed to be the water)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding box coordinates
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Draw a rectangle around the detected area on the original frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate the percentage of the bottle filled with water
        total_area = frame.shape[0] * frame.shape[1]
        water_area = cv2.contourArea(largest_contour)
        water_level_percentage = (water_area / (180*300)) * 100

        # Display the water level indicator inside the rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f'Water Level: {water_level_percentage:.2f}%'
        cv2.putText(frame, text, (x, y - 10), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Display the result
        cv2.imshow('Water Level', frame)

        return water_level_percentage, (x, y, w, h)

    return 0, None  # Return 0 if no contours are found (no water)

# Function to handle trackbar changes
def on_lower_color_change(val, index):
    global lower_color
    lower_color[index] = val

def on_upper_color_change(val, index):
    global upper_color
    upper_color[index] = val

# Open the webcam
cap = cv2.VideoCapture(0)

# Create a window and six trackbars for adjusting the HSV color values
cv2.namedWindow('Water Level')

# Initial lower and upper bounds for color in HSV (you can adjust these)
lower_color = np.array([8, 36, 63])
upper_color = np.array([31, 255, 255])

for i, color_name in enumerate(['Hue', 'Saturation', 'Value']):
    cv2.createTrackbar(f'Lower {color_name}', 'Water Level', lower_color[i], 255, lambda val, i=i: on_lower_color_change(val, i))
    cv2.createTrackbar(f'Upper {color_name}', 'Water Level', upper_color[i], 255, lambda val, i=i: on_upper_color_change(val, i))

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Get the current HSV values for color from the trackbars
    for i, color_name in enumerate(['Hue', 'Saturation', 'Value']):
        lower_color[i] = cv2.getTrackbarPos(f'Lower {color_name}', 'Water Level')
        upper_color[i] = cv2.getTrackbarPos(f'Upper {color_name}', 'Water Level')

    # Calculate and display the water level, and draw rectangle with indicator
    water_level_percentage, bounding_box = calculate_water_level(frame, lower_color, upper_color)
    print(f"Water level: {water_level_percentage:.2f}%")

    # If a bounding box is available, print its coordinates
    if bounding_box:
        x, y, w, h = bounding_box
        print(f"Bounding box coordinates: (x={x}, y={y}, w={w}, h={h})")

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
