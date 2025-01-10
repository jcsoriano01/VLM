import cv2
import requests
import base64
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def process_video_stream(server_url, video_source, task):
    
    if isinstance(video_source, int): #For webcam
        cap = cv2.VideoCapture(video_source)
    else:
        raise ValueError("Invalid video source")

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {video_source}")
    
    #try:
    with ThreadPoolExecutor(max_workers=2) as executor:
        #table_future = None
        frame_skip = 2  # Process every 2nd frame
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            #Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            frame_encoded = base64.b64encode(buffer).decode('utf-8')

            #Prepare payload with task and frame
            payload = {
                'prompt': task,
                'image': frame_encoded
            }

            # Send the frame to the server
            try:
                response = requests.post(server_url, json=payload, timeout=10)
                if response.status_code != 200:
                    print(f"Error: {response.status_code}")
                    continue
                response_data = response.json()
            except requests.exceptions.RequestException as e:
                print(f"Error sending frame: {str(e)}")
                continue

            processed_frame_data = base64.b64decode(frame_encoded)
            processed_frame = cv2.imdecode(
                np.frombuffer(processed_frame_data, np.uint8),
                cv2.IMREAD_COLOR,
            )

            _, processed_frame_buffer = cv2.imencode('.jpg', processed_frame)

            yield processed_frame_buffer.tobytes(), response_data

    # finally:
    cap.release()
        # print("Video stream closed.")