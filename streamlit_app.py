import streamlit as st
import torch
from PIL import Image
import numpy as np
from gtts import gTTS
import os
import base64
import time
import tempfile
import shutil

# Set page config for better accessibility
st.set_page_config(
    page_title="NavigateSolo - Seat Navigation",
    page_icon="🪑",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add custom CSS for better accessibility
st.markdown("""
<style>
    .main { padding: 1rem; }
    [data-testid="stCameraInput"] {
        width: 100%;
        max-width: 800px;
        margin: 0 auto;
    }
    [data-testid="stCameraInput"] button {
        width: 100% !important;
        height: 50px !important;
        font-size: 18px !important;
        margin-top: 10px !important;
    }
    /* Style for the switch camera button */
    [data-testid="stCameraInput"] button[aria-label="Switch camera"] {
        width: auto !important;
        position: absolute !important;
        top: 10px !important;
        right: 10px !important;
        background-color: rgba(255, 255, 255, 0.8) !important;
        border-radius: 50% !important;
        padding: 8px !important;
        min-width: 40px !important;
        height: 40px !important;
    }
    .accessibility-text {
        font-size: 1.2em;
        line-height: 1.5;
    }
    .help-button {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
    }
</style>
""", unsafe_allow_html=True)

# Debug: Verify cv2 import (only needed for debug mode)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Load pre-trained YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load("ultralytics/yolov5", "yolov5m", pretrained=True)

model = load_model()

# Create a temporary directory for audio files
@st.cache_resource
def get_temp_dir():
    temp_dir = tempfile.mkdtemp()
    return temp_dir

temp_dir = get_temp_dir()

# Cleanup function for temporary files
def cleanup_temp_files():
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

# Initialize session state
if 'temp_cleaned' not in st.session_state:
    cleanup_temp_files()
    st.session_state.temp_cleaned = True
if 'practice_mode' not in st.session_state:
    st.session_state.practice_mode = False
if 'haptic_feedback' not in st.session_state:
    st.session_state.haptic_feedback = True
if 'voice_guidance' not in st.session_state:
    st.session_state.voice_guidance = True
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

# Sidebar for accessibility settings
with st.sidebar:
    st.title("Accessibility Settings")
    st.session_state.practice_mode = st.toggle(
        "Practice Mode",
        st.session_state.practice_mode,
        help="In practice mode, the app will provide more detailed feedback and guidance"
    )
    st.session_state.haptic_feedback = st.toggle(
        "Haptic Feedback",
        st.session_state.haptic_feedback,
        help="Enable vibration patterns for distance feedback"
    )
    st.session_state.voice_guidance = st.toggle(
        "Voice Guidance",
        st.session_state.voice_guidance,
        help="Enable voice instructions"
    )
    
    st.markdown("---")
    st.markdown("### Quick Help")
    st.markdown("""
    - **Take Photo**: Press the large green button
    - **Repeat Instructions**: Press the button below the instructions
    - **Practice Mode**: Toggle in settings for detailed guidance
    - **Voice Command**: Say "help" for assistance
    - **Emergency Stop**: Say "stop" to pause guidance
    """)
    
    if st.toggle("Debug Mode", st.session_state.debug_mode):
        st.session_state.debug_mode = True
        st.markdown("Debug information enabled")

# Add help button with voice commands
st.markdown("""
<div class="help-button">
    <button onclick="speakHelp()" style="padding: 10px 20px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer;">
        Help
    </button>
</div>

<script>
    function speakHelp() {
        const helpText = "Welcome to NavigateSolo. To use the app, point your phone's camera and press the take photo button. The app will guide you to the nearest empty seat. You can say 'help' at any time for assistance, or 'stop' to pause guidance.";
        const utterance = new SpeechSynthesisUtterance(helpText);
        window.speechSynthesis.speak(utterance);
    }
    
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.continuous = true;
    recognition.interimResults = true;
    
    recognition.onresult = function(event) {
        const command = event.results[event.results.length - 1][0].transcript.toLowerCase();
        if (command.includes('help')) {
            speakHelp();
        } else if (command.includes('stop')) {
            window.speechSynthesis.cancel();
        }
    };
    
    recognition.start();
</script>
""", unsafe_allow_html=True)

# Add JavaScript for back camera
st.markdown("""
<script>
    async function setupCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: { exact: "environment" },
                    width: { ideal: 1920 },
                    height: { ideal: 1080 }
                }
            });
            const videoElement = document.querySelector('[data-testid="stCameraInput"] video');
            if (videoElement) {
                videoElement.srcObject = stream;
            }
        } catch (error) {
            console.error('Camera access error:', error);
            // Fallback to any available camera
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: true
                });
                const videoElement = document.querySelector('[data-testid="stCameraInput"] video');
                if (videoElement) {
                    videoElement.srcObject = stream;
                }
            } catch (fallbackError) {
                console.error('Fallback camera access error:', fallbackError);
            }
        }
    }
    // Call setupCamera when the page loads
    window.addEventListener('load', setupCamera);
    // Also try to set up camera immediately
    setupCamera();
</script>
""", unsafe_allow_html=True)

# Title and instructions
st.title("NavigateSolo - Seat Navigation")
st.markdown('<div class="accessibility-text">Welcome to NavigateSolo. Point your phone\'s camera and press the take photo button. I will guide you to the nearest empty seat with clear instructions.</div>', unsafe_allow_html=True)

# Function to get spatial description
def get_spatial_description(chair, img_width, img_height):
    center_x = (chair['xmin'] + chair['xmax']) / 2
    center_y = (chair['ymin'] + chair['ymax']) / 2
    
    # Horizontal position
    if center_x < img_width * 0.3:
        horizontal = "far to your left"
    elif center_x < img_width * 0.4:
        horizontal = "to your left"
    elif center_x > img_width * 0.7:
        horizontal = "far to your right"
    elif center_x > img_width * 0.6:
        horizontal = "to your right"
    else:
        horizontal = "straight ahead"
    
    # Add vertical position in practice mode
    if st.session_state.practice_mode:
        if center_y < img_height * 0.3:
            vertical = "in the front of the room"
        elif center_y > img_height * 0.7:
            vertical = "in the back of the room"
        else:
            vertical = "in the middle of the room"
        return f"{horizontal}, {vertical}"
    
    return horizontal

# Function to get distance description
def get_distance_description(area):
    if area > 12000:
        return {"description": "very close to you", "steps": 2, "haptic": "short"}
    elif area > 10000:
        return {"description": "close to you", "steps": 4, "haptic": "medium"}
    elif area > 7000:
        return {"description": "at a moderate distance", "steps": 7, "haptic": "long"}
    elif area > 5000:
        return {"description": "far from you", "steps": 10, "haptic": "very_long"}
    else:
        return {"description": "very far from you", "steps": 15, "haptic": "continuous"}

# Function to generate audio instructions
def generate_audio(distance_info, chair, img_width, img_height):
    spatial_desc = get_spatial_description(chair, img_width, img_height)
    
    if st.session_state.practice_mode:
        message = f"I can see an empty seat {spatial_desc}. It's {distance_info['description']}, about {distance_info['steps']} steps away. Walk slowly and take another picture when you're ready for an update. Remember, you can say 'help' for assistance or 'stop' to pause guidance."
    else:
        message = f"Empty seat {spatial_desc}, {distance_info['description']}. About {distance_info['steps']} steps. Walk slowly and take another picture for an update."
    
    tts = gTTS(text=message, lang="en", slow=False)
    audio_file = os.path.join(temp_dir, f"instructions_{int(time.time())}.mp3")
    tts.save(audio_file)
    
    # Generate haptic feedback if enabled
    if st.session_state.haptic_feedback:
        haptic_pattern = distance_info['haptic']
        st.markdown(f"""
        <script>
            if (navigator.vibrate) {{
                const pattern = {{
                    'short': [100, 100, 100],
                    'medium': [200, 100, 200],
                    'long': [300, 100, 300],
                    'very_long': [400, 100, 400],
                    'continuous': [500, 100, 500, 100, 500]
                }}['{haptic_pattern}'];
                navigator.vibrate(pattern);
            }}
        </script>
        """, unsafe_allow_html=True)
    
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

# Camera input with error handling
try:
    picture = st.camera_input("Take a picture", key="camera_input")
except Exception as e:
    st.error("Camera access error. Please ensure camera permissions are granted and try again.")
    st.stop()

# Process the image
if picture is not None:
    with st.spinner("Processing image..."):
        try:
            # Convert image and run inference
            img = Image.open(picture)
            img_height, img_width = img.height, img.width
            
            results = model(img)
            detections = results.pandas().xyxy[0]
            
            # Filter detections with confidence threshold
            confidence_threshold = 0.5
            chairs = detections[(detections['name'] == 'chair') & (detections['confidence'] > confidence_threshold)]
            people = detections[(detections['name'] == 'person') & (detections['confidence'] > confidence_threshold)]
            belongings = detections[(detections['name'].isin(['backpack', 'handbag', 'suitcase', 'book', 'laptop'])) & 
                                  (detections['confidence'] > confidence_threshold)]

            # Find empty chairs
            empty_chairs = []
            for _, chair in chairs.iterrows():
                chair_box = [chair['xmin'], chair['ymin'], chair['xmax'], chair['ymax']]
                area = (chair['xmax'] - chair['xmin']) * (chair['ymax'] - chair['ymin'])
                is_occupied = False

                # Check for person overlap
                for _, person in people.iterrows():
                    person_box = [person['xmin'], person['ymin'], person['xmax'], person['ymax']]
                    if (chair_box[0] < person_box[2] and chair_box[2] > person_box[0] and
                        chair_box[1] < person_box[3] and chair_box[3] > person_box[1]):
                        is_occupied = True
                        break

                # Check for belongings
                if not is_occupied:
                    for _, belonging in belongings.iterrows():
                        belonging_box = [belonging['xmin'], belonging['ymin'], belonging['xmax'], belonging['ymax']]
                        if (chair_box[0] < belonging_box[2] and chair_box[2] > belonging_box[0] and
                            chair_box[1] < belonging_box[3] and chair_box[3] > belonging_box[1]):
                            is_occupied = True
                            break

                if not is_occupied:
                    empty_chairs.append({"chair": chair, "area": area, "ymax": chair['ymax']})

            # Provide guidance for closest empty chair
            if empty_chairs:
                # Select chair with highest ymax (closest to camera)
                closest_chair = max(empty_chairs, key=lambda x: x["ymax"])
                distance_info = get_distance_description(closest_chair["area"])
                audio_file, message = generate_audio(distance_info, closest_chair["chair"], img_width, img_height)
                
                if st.session_state.voice_guidance:
                    autoplay_audio(audio_file)
                
                st.write(message)
                
                # Add repeat instructions button
                if st.button("🔊 Repeat Instructions", key="repeat"):
                    autoplay_audio(audio_file)
                    st.write(message)

                # Trigger haptic feedback
                if st.session_state.haptic_feedback:
                    haptic_pattern = distance_info['haptic']
                    st.markdown(f"""
                    <script>
                        if (navigator.vibrate) {{
                            const pattern = {{
                                'short': [100, 100, 100],
                                'medium': [200, 100, 200],
                                'long': [300, 100, 300],
                                'very_long': [400, 100, 400],
                                'continuous': [500, 100, 500, 100, 500]
                            }}['{haptic_pattern}'];
                            navigator.vibrate(pattern);
                        }}
                    </script>
                    """, unsafe_allow_html=True)
            else:
                no_seat_message = "I don't see any empty seats in the current view. Please try taking another picture from a different angle."
                if st.session_state.voice_guidance:
                    tts = gTTS(text=no_seat_message, lang="en", slow=False)
                    audio_file = os.path.join(temp_dir, f"no_seats_{int(time.time())}.mp3")
                    tts.save(audio_file)
                    autoplay_audio(audio_file)
                
                st.write(no_seat_message)
                
                if st.button("🔊 Repeat Instructions", key="repeat"):
                    autoplay_audio(audio_file)
                    st.write(no_seat_message)

            # Always show visualization if OpenCV is available
            if CV2_AVAILABLE:
                img_array = np.array(img)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Draw all chairs
                for _, chair in chairs.iterrows():
                    xmin, ymin, xmax, ymax = map(int, [chair['xmin'], chair['ymin'], chair['xmax'], chair['ymax']])
                    is_empty = any(c["chair"].equals(chair) for c in empty_chairs)
                    color = (0, 255, 0) if is_empty else (0, 0, 255)
                    cv2.rectangle(img_array, (xmin, ymin), (xmax, ymax), color, 2)
                    
                    if st.session_state.debug_mode:
                        cv2.putText(img_array, f"{chair['confidence']:.2f}", (xmin, ymin - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Highlight closest empty chair
                if empty_chairs:
                    chair = closest_chair["chair"]
                    xmin, ymin, xmax, ymax = map(int, [chair['xmin'], chair['ymin'], chair['xmax'], chair['ymax']])
                    cv2.rectangle(img_array, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

                st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), caption="Chair Detection", use_container_width=True)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("Please try taking another picture. Make sure the image is clear and well-lit.")
            if st.session_state.debug_mode:
                st.exception(e)