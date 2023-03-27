import cv2
import numpy as np
import time

# Initialize the motion history image
cap = cv2.VideoCapture(0)
height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
mhi = np.zeros((height, width), dtype=np.float32)

# Initialize the previous frame, frame count, and time
gray_prev = None
frame_count = 1
start_time = time.time()
seconds = 5

# Define the color map
hsv_map = np.zeros((256, 1, 3), dtype=np.uint8)
print(hsv_map)
hsv_map[:, :, 0] = np.arange(0, 256).reshape(-1, 1)
hsv_map[:, :, 1] = 255
hsv_map[:, :, 2] = 255
print(hsv_map)
color_map = cv2.cvtColor(hsv_map, cv2.COLOR_HSV2BGR)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute the binary motion mask
    if gray_prev is not None:
        diff = cv2.absdiff(gray, gray_prev)
        ret, motion_mask = cv2.threshold(diff, 4, 1, cv2.THRESH_BINARY)
        timestamp = time.time() - start_time
        cv2.motempl.updateMotionHistory(motion_mask, mhi, timestamp, 1)

    # Normalize the motion history image
    if frame_count > 0:
        mhi_norm = np.uint8(np.clip((mhi - (frame_count - 1) / 2) / ((frame_count - 1) / 2), 0, 1) * 255)
    else:
        mhi_norm = np.zeros_like(mhi)
    # print(np.unique(mhi_norm))
    # print(np.unique(mhi))

    # Apply the color map
    mhi_color = cv2.applyColorMap(mhi_norm, color_map)

    # Show the resulting image
    cv2.imshow('Motion history image', mhi_color)

    # Update the previous frame and frame count
    gray_prev = gray.copy()
    frame_count += 1

    # Check if the specified number of seconds has elapsed
    elapsed_time = time.time() - start_time
    if elapsed_time >= seconds:
        break

    # Wait for a key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()