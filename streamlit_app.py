import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np

# Load pre-trained YOLOv5 model (yolov5m)
@st.cache_resource
def load_model():
    model = torch.hub.load("ultralytics/yolov5", "yolov5m", pretrained=True)
    return model

model = load_model()

# Title of the app
st.title("Live Seat Detection and Navigation App with YOLOv5")

# Instructions
st.write("Use your webcam to take a picture, and the app will detect chairs and guide you to the nearest empty seat.")

# Camera input widget
picture = st.camera_input("Take a picture")

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

# Process the image with YOLOv5
if picture is not None:
    # Display the captured image
    st.image(picture, caption="Captured Image", use_container_width=True)

    # Convert Streamlit's BytesIO to PIL Image
    img = Image.open(picture)

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

    # Display chair status
    if not chairs.empty:
        st.write("Chair Status:")
        for chair_idx, status in chair_status.items():
            chair = chairs.loc[chair_idx]
            st.write(f"- Chair at ({int(chair['xmin'])}, {int(chair['ymin'])}): {status} (Confidence: {chair['confidence']:.2f})")
    else:
        st.write("No chairs detected in the image.")
        st.stop()

    # Find the nearest empty chair
    # Assume user's position is at the bottom center of the image
    img_width, img_height = img.size
    user_position = [img_width / 2, img_height]  # Bottom center

    nearest_empty_chair = None
    nearest_distance = float('inf')
    nearest_chair_idx = None

    empty_chairs = {idx: status for idx, status in chair_status.items() if status == "Empty"}
    for chair_idx in empty_chairs:
        chair = chairs.loc[chair_idx]
        chair_center = [(chair['xmin'] + chair['xmax']) / 2, (chair['ymin'] + chair['ymax']) / 2]
        distance = np.sqrt((chair_center[0] - user_position[0]) ** 2 + (chair_center[1] - user_position[1]) ** 2)
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_empty_chair = chair_center
            nearest_chair_idx = chair_idx

    # Provide navigation instructions
    if nearest_empty_chair is not None:
        st.write("### Navigation Instructions:")
        dx = nearest_empty_chair[0] - user_position[0]
        dy = nearest_empty_chair[1] - user_position[1]

        # Determine direction
        direction = "Move forward"
        if abs(dx) > img_width * 0.1:  # Threshold for left/right movement
            if dx > 0:
                direction += " and slightly right"
            else:
                direction += " and slightly left"

        st.write(f"{direction} to reach the nearest empty chair at position ({int(nearest_empty_chair[0])}, {int(nearest_empty_chair[1])}).")
    else:
        st.write("No empty chairs detected in the image.")
        st.stop()

    # Render image with custom labels using PIL
    img_array = np.array(img)  # PIL Image to numpy array (RGB)
    img = Image.fromarray(img_array)  # Convert back to PIL Image
    draw = ImageDraw.Draw(img)

    # Try to use a default font, fall back to no font if unavailable
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = None

    for chair_idx, status in chair_status.items():
        chair = chairs.loc[chair_idx]
        xmin, ymin, xmax, ymax = int(chair['xmin']), int(chair['ymin']), int(chair['xmax']), int(chair['ymax'])
        # Highlight nearest empty chair in blue, others in green/red
        if chair_idx == nearest_chair_idx and status == "Empty":
            color = (0, 0, 255)  # Blue for nearest empty chair
            label = "Nearest Empty"
        else:
            color = (0, 255, 0) if status == "Empty" else (255, 0, 0)  # Green for empty, red for occupied
            label = status

        # Draw rectangle
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
        # Draw text
        text_position = (xmin, ymin - 30 if ymin >= 30 else ymin + 30)
        draw.text(text_position, label, fill=color, font=font)

    # Display the image with detections
    st.image(img, caption="Image with Chair Status and Navigation", use_container_width=True)