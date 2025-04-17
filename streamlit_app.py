import streamlit as st
import torch
from PIL import Image
import io
import numpy as np
from gtts import gTTS
import os
import base64
import time

# Debug: Verify cv2 import
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as e:
    st.warning(f"OpenCV import failed: {e}. Visualization disabled, but navigation will work.")
    CV2_AVAILABLE = False

# Load pre-trained YOLOv5 model (yolov5s for faster inference)
@st.cache_resource
def load_model():
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    return model

model = load_model()

# Title of the app
st.title("Seat Detection")

# Instructions
st.write("The app will automatically detect empty chairs and provide audio instructions. Use the button to start/stop live detection.")

# Initialize session state
if "live_detection" not in st.session_state:
    st.session_state.live_detection = False
if "last_audio_time" not in st.session_state:
    st.session_state.last_audio_time = 0
if "last_message" not in st.session_state:
    st.session_state.last_message = ""
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None
if "frame_key" not in st.session_state:
    st.session_state.frame_key = 0

# Toggle live detection
if st.button("Start/Stop Live Detection", key="toggle_live"):
    st.session_state.live_detection = not st.session_state.live_detection

# Camera input widget
picture = st.camera_input("Live Feed", key=f"camera_{st.session_state.frame_key}")

# Function to calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# Function to calculate distance between box centers
def calculate_center_distance(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    center1_x, center1_y = (x1 + x2) / 2, (y1 + y2) / 2
    center2_x, center2_y = (x3 + x4) / 2, (y3 + y4) / 2
    return np.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)

# Function to estimate distance from bounding box area
def estimate_distance(area):
    if area > 2000:
        return {"range": "near", "steps": 3}
    elif area > 500:
        return {"range": "medium", "steps": 7}
    else:
        return {"range": "far", "steps": 12}

# Function to determine direction (left or right)
def get_direction(chair, img_width):
    center_x = (chair['xmin'] + chair['xmax']) / 2
    if center_x < img_width / 2:
        return "to the left"
    else:
        return "to the right"

# Function to generate audio instructions
def generate_audio(distance_info, direction):
    message = f"The nearest empty seat is {distance_info['range']}, about {distance_info['steps']} steps ahead {direction}. Walk straight and keep the camera steady."
    tts = gTTS(text=message, lang="en")
    audio_file = f"instructions_{int(time.time())}.mp3"
    tts.save(audio_file)
    return audio_file, message

# Function to auto-play audio
def autoplay_audio(audio_file):
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    b64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
    <audio autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# Process the image with YOLOv5
if picture is not None and st.session_state.live_detection:
    # Convert Streamlit's BytesIO to PIL Image
    img = Image.open(picture)
    img_height = img.height  # Get image height for bottom threshold
    img_width = img.width    # Get image width for direction

    # Run inference
    results = model(img)
    detections = results.pandas().xyxy[0]

    # Filter detections
    chairs = detections[detections['name'] == 'chair']
    people = detections[detections['name'] == 'person']
    belongings = detections[detections['name'].isin(['backpack', 'handbag', 'suitcase', 'book', 'laptop'])]

    # Classify chairs as empty or occupied
    chair_status = {}
    for chair_idx, chair in chairs.iterrows():
        chair_box = [chair['xmin'], chair['ymin'], chair['xmax'], chair['ymax']]
        is_occupied = False

        # Check for person overlap (sitting) or proximity
        for _, person in people.iterrows():
            person_box = [person['xmin'], person['ymin'], person['xmax'], person['ymax']]
            iou = calculate_iou(chair_box, person_box)
            distance = calculate_center_distance(chair_box, person_box)
            chair_width = chair_box[2] - chair_box[0]
            if iou > 0.3 or (distance < chair_width * 1.5 and iou > 0.1):
                is_occupied = True
                break

        # Check for belongings overlap
        if not is_occupied:
            for _, belonging in belongings.iterrows():
                belonging_box = [belonging['xmin'], belonging['ymin'], belonging['xmax'], belonging['ymax']]
                iou = calculate_iou(chair_box, belonging_box)
                if iou > 0.3:
                    is_occupied = True
                    break

        chair_status[chair_idx] = "Occupied" if is_occupied else "Empty"

    # Navigation: Find closest empty chair
    empty_chairs = []
    for chair_idx, status in chair_status.items():
        if status == "Empty":
            chair = chairs.loc[chair_idx]
            area = (chair['xmax'] - chair['xmin']) * (chair['ymax'] - chair['ymin'])
            ymax = chair['ymax']
            empty_chairs.append({"idx": chair_idx, "area": area, "ymax": ymax, "chair": chair})

    current_time = time.time()
    if empty_chairs:
        # Define bottom threshold (within 20% of image height from bottom)
        bottom_threshold = img_height * 0.8
        # Filter chairs near bottom
        bottom_chairs = [c for c in empty_chairs if c["ymax"] >= bottom_threshold]
        
        if bottom_chairs:
            # Select chair with largest ymax (closest to bottom)
            closest_chair = max(bottom_chairs, key=lambda x: x["ymax"])
        else:
            # Fallback: Largest area among chairs with highest ymax
            closest_chair = max(empty_chairs, key=lambda x: (x["ymax"], x["area"]))

        # Get direction and estimate distance
        direction = get_direction(closest_chair["chair"], img_width)
        distance_info = estimate_distance(closest_chair["area"])
        audio_file, message = generate_audio(distance_info, direction)
        
        # Auto-play audio if new instruction (every 3 seconds)
        if current_time - st.session_state.last_audio_time > 3 or message != st.session_state.last_message:
            autoplay_audio(audio_file)
            st.write(message)  # For screen readers
            st.session_state.last_audio = audio_file
            st.session_state.last_message = message
            st.session_state.last_audio_time = current_time
    else:
        no_seat_message = "No empty seats found. Please adjust the camera."
        if current_time - st.session_state.last_audio_time > 3 or no_seat_message != st.session_state.last_message:
            tts = gTTS(text=no_seat_message, lang="en")
            audio_file = f"no_seats_{int(time.time())}.mp3"
            tts.save(audio_file)
            autoplay_audio(audio_file)
            st.write(no_seat_message)
            st.session_state.last_audio = audio_file
            st.session_state.last_message = no_seat_message
            st.session_state.last_audio_time = current_time

    # Display chair status (for debugging or sighted users)
    if not chairs.empty:
        st.write("Chair Status:")
        for chair_idx, status in chair_status.items():
            chair = chairs.loc[chair_idx]
            st.write(f"- Chair at ({int(chair['xmin'])}, {int(chair['ymin'])}): {status} (Confidence: {chair['confidence']:.2f})")
    else:
        st.write("No chairs detected in the frame.")

    # Render image with custom labels (if OpenCV is available)
    if CV2_AVAILABLE:
        img_array = np.array(img)  # PIL Image to numpy array (RGB)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for cv2
        for chair_idx, status in chair_status.items():
            chair = chairs.loc[chair_idx]
            xmin, ymin, xmax, ymax = int(chair['xmin']), int(chair['ymin']), int(chair['xmax']), int(chair['ymax'])
            color = (0, 0, 255) if status == "Occupied" else (0, 255, 0)  # Red for occupied, green for empty
            cv2.rectangle(img_array, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(img_array, status, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Highlight closest empty chair (if any)
        if empty_chairs:
            chair = closest_chair["chair"]
            direction = get_direction(chair, img_width)
            xmin, ymin, xmax, ymax = int(chair['xmin']), int(chair['ymin']), int(chair['xmax']), int(chair['ymax'])
            cv2.putText(img_array, f"Closest Empty ({direction})", (xmin, ymin - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the image with detections
        st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), caption="Live Feed with Chair Status", use_container_width=True)

    # Refresh camera input for next frame
    st.session_state.frame_key += 1
    st.rerun()

# Repeat last audio instructions
if "last_audio" in st.session_state:
    if st.button("Repeat Last Instructions", key="repeat"):
        autoplay_audio(st.session_state.last_audio)
        st.write(st.session_state.last_message)