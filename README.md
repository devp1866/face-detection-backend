# Face Shape Analysis API

This is a Flask-based API that analyzes face shapes using machine learning. It uses MediaPipe for face landmark detection and a pre-trained Random Forest model to classify face shapes into different categories.

## Features

- Real-time face shape detection using webcam
- Face shape analysis from uploaded images
- Integration with Cloudinary for image storage
- MongoDB for storing analysis results
- Support for multiple face shape classifications (Heart, Oval, Round, Square)

## Prerequisites

- Python 3.12+
- MongoDB
- Cloudinary account
- Webcam (for real-time detection)

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd face-back
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your credentials:
```
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
MONGO_URI=your_mongodb_connection_string
```

4. Download the MediaPipe face landmarker model:
```bash
wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task -O face_landmarker_v2_with_blendshapes.task
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. The API will be available at `http://localhost:5000`

## API Endpoints

- `GET /` - Home page
- `POST /analyze` - Analyze face shape from uploaded image
- `GET /video_feed` - Real-time video feed with face shape analysis
- `GET /real_time` - Real-time face shape detection page

## Model Information

The face shape classification uses a Random Forest model trained on facial measurements and landmarks. The model is stored in `Best_RandomForest.pkl`.

## Contributing

Feel free to open issues and pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 