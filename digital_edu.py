import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.pt') # Load a pretrained YOLOv8n model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Load the Haar Cascade for face detection
cap = cv2.VideoCapture(0) # Open the default camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)# Set the width of the camera frame
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # Set the height of the camera frame


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1) # Flip the frame horizontally for a mirror effect
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert the frame to grayscale for face detection
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80)) # Detect faces in the frame
    for (x, y, w, h) in faces:
        px = int(w * 0.12)
        py = int(h * 0.12)
        x1 = max(0, x - px)
        y1 = max(0, y - py)
        x2 = min(frame.shape[1], x + w + px)
        y2 = min(frame.shape[0], y + h + py)
        face = frame[y1:y2, x1:x2] # Extract the detected face region from the frame    

        if face.size > 0:
            smooth = cv2.bilateralFilter(face, 11, 80, 80) # Apply bilateral filter for smoothing
            hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV) # Convert the smoothed face region to HSV color space
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.08, 0, 255) # Increase the saturation channel to enhance colors
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.06 + 3, 0, 255) # Increase the value channel to enhance brightness
            smooth = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) # Convert the enhanced HSV image back to BGR color space
            b, g, r = cv2.split(smooth) # Split the enhanced face region into its BGR channels
            r = np.clip(r.astype(np.float32) + 3, 0, 255).astype(np.uint8) # Increase the red channel to enhance skin tone

            smooth = cv2.merge((b, g, r)) # Merge the enhanced BGR channels back together
            fh, fw = face.shape[:2]
            mask = np.zeros((fh, fw), dtype=np.uint8) # Create a mask for the face region
            cv2.ellipse(mask, (fw // 2, fh // 2), (int(fw * 0.43), int(fh * 0.47)), 0, 0, 360, 255, -1) # Draw an ellipse on the mask to define the face area
            mask = cv2.GaussianBlur(mask, (0, 0), 15) # Apply Gaussian blur to the mask for smooth edges
            mask = mask.astype(float) / 255.0 # Normalize the mask to the range [0, 1]
            mask = mask[..., None] * 0.52
            result = (face * (1 - mask) + smooth * mask).astype(np.uint8) # Blend the original face region with the enhanced face region using the mask
            frame[y1:y2, x1:x2] = result # Replace the original face region in the frame with the enhanced face region

    results = model(frame, stream=True) # Perform object detection on the frame
    for r in results:
        frame = r.plot() # Draw the detection results on the frame
    cv2.imshow('Real-Time Object Detection + Beauty Filter', frame) # Display the frame with detection results
    if cv2.waitKey(1) & 0xFF == ord('q'): # Exit the loop if 'q' is pressed
        break
cap.release() # Release the camera
cv2.destroyAllWindows() # Close all OpenCV windows

