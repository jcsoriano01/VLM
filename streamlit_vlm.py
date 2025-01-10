import streamlit as st
from client_vlm import process_video_stream
import cv2
import numpy as np

# Function to detect available cameras
def get_available_cameras(max_cameras=5):
    cameras = []
    for index in range(max_cameras):
        cap = cv2.VideoCapture(index)
        if cap.isOpened() and cap.read()[0]:
            cameras.append(index)
            cap.release()
    return cameras

# Set page configuration
st.set_page_config(
    page_title="Vision Language Model",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="auto",
)

# Page Title
st.markdown(
    """
    <h1 style="text-align: left; color:#4a90e2; font-size: 32px; margin-top: 0px;">
        Vision Language Model
    </h1>
    """,
    unsafe_allow_html=True,
)

# Sidebar Configuration
st.sidebar.header("Configuration")

# Add an input bar
user_prompt = st.sidebar.text_input("Enter your prompt:", placeholder="Type your prompt here...")

cameras = get_available_cameras()
if not cameras:
    st.sidebar.error("No cameras detected.")
    st.stop()
    
selected_camera = st.sidebar.selectbox(
    "Select Camera", 
    cameras, 
    format_func=lambda x: f"Camera {x}"  # Display as "Camera 0", "Camera 1", etc.
)

start_detection = st.sidebar.button("Start Detecting", disabled=not (user_prompt and selected_camera is not None))

if start_detection:
    # Ensure prompt and camera selection are valid
    if not user_prompt:
        st.error("Please enter a prompt before starting inference.")
        st.stop()

    if selected_camera is None:
        st.error("Please select a camera before starting inference.")
        st.stop()
    
    video_source = selected_camera

    cap = cv2.VideoCapture(selected_camera)
    if not cap.isOpened():
        st.error(f"Camera {selected_camera} could not be opened.")
        st.stop()

    server_url = "http://127.0.0.1:5050/predict"
    # Split the layout into two columns of equal width
    col1 = st.columns(1)[0]

    # Placeholders for webcam and table
    webcam_placeholder = col1.empty()

    # Placeholders for FPS and detection messages
    infer_placeholder = col1.empty()
    video_display_height = 300
    # Process video stream
    for frame_bytes, infer in process_video_stream(server_url, video_source, user_prompt):
        if frame_bytes:
            current_frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)

            # Resize the frame to the desired height while keeping the aspect ratio
            aspect_ratio = current_frame.shape[1] / current_frame.shape[0]
            new_width = int(video_display_height * aspect_ratio)
            resized_frame = cv2.resize(current_frame, (new_width, video_display_height))

            # Display video frame
            webcam_placeholder.image(
                resized_frame,
                channels="BGR",
                use_column_width=True
            )
        # Update inference results
        infer_placeholder.markdown(
            f"<h3 style='text-align: left; color:black; font-size: 18px;'>Inference: {infer.get("response", "Error in response")}</h3>",
            unsafe_allow_html=True
        )
