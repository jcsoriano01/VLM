import cv2
import base64
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from vlm_infer_js import process_frame  # Import the function from the modified server.py

def process_video_stream(video_source, task):
    if isinstance(video_source, int):  # For webcam
        cap = cv2.VideoCapture(video_source)
    else:
        raise ValueError("Invalid video source")

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {video_source}")

    with ThreadPoolExecutor(max_workers=2) as executor:
        frame_skip = 2  # Process every 2nd frame
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            frame_encoded = base64.b64encode(buffer).decode('utf-8')

            # Call the server function to process the frame
            try:
                response = process_frame(base64.b64decode(frame_encoded), task)
                print(f"Server response: {response}")

                # You can handle the response here if needed (e.g., displaying, saving results)
                # Example: Display processed frame (optional, just for visualization)
                # decoded_frame = cv2.imdecode(np.frombuffer(base64.b64decode(frame_encoded), np.uint8), cv2.IMREAD_COLOR)
                # cv2.imshow("Processed Frame", decoded_frame)

            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                continue

    cap.release()
