from flask import Flask, request, render_template, Response, jsonify
import cv2
import numpy as np
import pickle
import os
import warnings
from werkzeug.utils import secure_filename
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from flask import Flask, request, render_template, Response, url_for, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
# Import database functions
from database import upload_image_to_cloudinary, save_analysis_to_db

# Load environment variables
load_dotenv()

# Suppress specific deprecation warnings from protobuf
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')
app = Flask(__name__, template_folder='templates')
CORS(app)  # Enable CORS for all routes

# Initialize MediaPipe Face Landmarker
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

# Load the pre-trained model (for inference)
with open('Best_RandomForest.pkl', 'rb') as f:
    face_shape_model = pickle.load(f)

def distance_3d(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_face_features(coords):
    # Define indices for landmarks
    landmark_indices = {
        'forehead': 10,
        'chin': 152,
        'left_cheek': 234,
        'right_cheek': 454,
        'left_eye': 263,
        'right_eye': 33,
        'nose_tip': 1
    }

    # Extract features based on landmark indices
    features = []
    landmarks_dict = {name: coords[idx] for name, idx in landmark_indices.items()}

    # Calculate distances between important landmarks
    features.append(distance_3d(landmarks_dict['forehead'], landmarks_dict['chin']))  # Face height
    features.append(distance_3d(landmarks_dict['left_cheek'], landmarks_dict['right_cheek']))  # Face width
    features.append(distance_3d(landmarks_dict['left_eye'], landmarks_dict['right_eye']))  # Eye distance

    # Additional distances
    features.append(distance_3d(landmarks_dict['nose_tip'], landmarks_dict['left_eye']))  # Nose to left eye
    features.append(distance_3d(landmarks_dict['nose_tip'], landmarks_dict['right_eye']))  # Nose to right eye
    features.append(distance_3d(landmarks_dict['chin'], landmarks_dict['left_cheek']))  # Chin to left cheek
    features.append(distance_3d(landmarks_dict['chin'], landmarks_dict['right_cheek']))  # Chin to right cheek
    features.append(distance_3d(landmarks_dict['forehead'], landmarks_dict['left_eye']))  # Forehead to left eye
    features.append(distance_3d(landmarks_dict['forehead'], landmarks_dict['right_eye']))  # Forehead to right eye

    # Additional features

    # # Facial aspect ratios
    # face_width = distance_3d(landmarks_dict['left_cheek'], landmarks_dict['right_cheek'])
    # face_height = distance_3d(landmarks_dict['forehead'], landmarks_dict['chin'])
    # eye_distance = distance_3d(landmarks_dict['left_eye'], landmarks_dict['right_eye'])

    # features.append(face_width / face_height)  # Aspect ratio of face width to height
    # features.append(face_height / eye_distance)  # Aspect ratio of face height to eye distance

    # # More distance features
    # features.append(distance_3d(landmarks_dict['left_eye'], landmarks_dict['chin']))  # Eye to chin
    # features.append(distance_3d(landmarks_dict['right_eye'], landmarks_dict['chin']))  # Eye to chin
    # features.append(distance_3d(landmarks_dict['left_cheek'], landmarks_dict['forehead']))  # Cheek to forehead
    # features.append(distance_3d(landmarks_dict['right_cheek'], landmarks_dict['forehead']))  # Cheek to forehead

    return np.array(features)

def get_face_shape_label(label):
    shapes = ["Heart", "Oval", "Round", "Square"]
    return shapes[label]

# Initialize MediaPipe Face Landmarker
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define function to compute Euclidean distance in 3D
def distance_3d(p1, p2):
    return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Create landmark proto
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        # Draw face landmarks
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

    return annotated_image



def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_frames():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = face_landmarker.detect(image)

            if detection_result.face_landmarks:
                for face_landmarks in detection_result.face_landmarks:
                    landmarks = [[lm.x, lm.y, lm.z] for lm in face_landmarks]
                    landmarks = np.array(landmarks)
                    face_features = calculate_face_features(landmarks)
                    face_shape_label = face_shape_model.predict([face_features])[0]
                    face_shape = get_face_shape_label(face_shape_label)
                    annotated_image = draw_landmarks_on_image(rgb_frame, detection_result)
                    cv2.putText(annotated_image, f"Face Shape: {face_shape}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                annotated_image = rgb_frame

            ret, buffer = cv2.imencode('.jpg', annotated_image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_face():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # --- 1. Read image and perform analysis first ---
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        detection_result = face_landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            return jsonify({"error": "No face detected"}), 400

        # --- 2. Get data, calculate features, and predict shape ---
        face_landmarks = detection_result.face_landmarks[0]
        
        # First, calculate the features
        landmarks_normalized = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks])
        face_features = calculate_face_features(landmarks_normalized)
        
        # Then, predict the shape
        face_shape_label = face_shape_model.predict([face_features])[0]
        face_shape = get_face_shape_label(face_shape_label)

        # --- 3. Draw landmarks on the image ---
        annotated_image_rgb = draw_landmarks_on_image(rgb_image, detection_result)
        cv2.putText(annotated_image_rgb, f"Face Shape: {face_shape}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # --- 4. Upload the PROCESSED image to Cloudinary ---
        annotated_image_bgr = cv2.cvtColor(annotated_image_rgb, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', annotated_image_bgr)
        processed_image_url = upload_image_to_cloudinary(buffer.tobytes())
        
        if not processed_image_url:
            return jsonify({"error": "Failed to upload processed image"}), 500

        # --- 5. Calculate Measurements using IPD calibration ---
        landmarks_normalized = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks])

        # Define more accurate landmark points for measurements
        # As per official MediaPipe documentation for better anatomical representation
        p_iris_l = landmarks_normalized[473] # Left Iris
        p_iris_r = landmarks_normalized[468] # Right Iris
        
        p_forehead_top = landmarks_normalized[10]  # Top of forehead hairline
        p_chin_tip = landmarks_normalized[152]     # Bottom of chin

        p_cheek_l = landmarks_normalized[234]      # Left cheekbone edge
        p_cheek_r = landmarks_normalized[454]      # Right cheekbone edge

        p_jaw_l = landmarks_normalized[172]        # Left jaw point
        p_jaw_r = landmarks_normalized[397]        # Right jaw point

        p_forehead_l = landmarks_normalized[63]   # Left forehead edge
        p_forehead_r = landmarks_normalized[293]  # Right forehead edge
        
        # IPD-based calibration
        AVG_IPD_CM = 6.3
        dist_iris = distance_3d(p_iris_l, p_iris_r)
        cm_per_unit = AVG_IPD_CM / dist_iris if dist_iris != 0 else 0

        # Calculate all distances
        dist_face_length = distance_3d(p_forehead_top, p_chin_tip)
        dist_cheek_width = distance_3d(p_cheek_l, p_cheek_r)
        dist_jaw_width = distance_3d(p_jaw_l, p_jaw_r)
        dist_forehead_width = distance_3d(p_forehead_l, p_forehead_r)

        # Convert to cm
        face_length_cm = dist_face_length * cm_per_unit
        cheekbone_width_cm = dist_cheek_width * cm_per_unit
        jaw_width_cm = dist_jaw_width * cm_per_unit
        forehead_width_cm = dist_forehead_width * cm_per_unit
        
        # Jaw curve ratio is a relative measure, so it doesn't need cm conversion
        jaw_curve_ratio = dist_face_length / dist_cheek_width if dist_cheek_width != 0 else 0

        measurements = {
            "face_length_cm": float(face_length_cm)-4,
            "cheekbone_width_cm": float(cheekbone_width_cm)+3,
            "jaw_width_cm": float(jaw_width_cm),
            "forehead_width_cm": float(forehead_width_cm)+3,
            "jaw_curve_ratio": float(jaw_curve_ratio)
        }

        # --- 6. Save analysis to MongoDB and return ---
        analysis_id = save_analysis_to_db(processed_image_url, face_shape, measurements)
        if not analysis_id:
            return jsonify({"error": "Failed to save analysis"}), 500
        
        # --- 7. Return the complete result ---
        return jsonify({
            "message": "Analysis successful",
            "analysis_id": analysis_id,
            "image_url": processed_image_url, # This is now the URL of the annotated image
            "face_shape": face_shape,
            "measurements": measurements
        })

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/real_time')
def real_time():
    return render_template('real_time.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
