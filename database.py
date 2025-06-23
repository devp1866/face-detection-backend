import os
import cloudinary
import cloudinary.uploader
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# --- Cloudinary Configuration ---
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# --- MongoDB Configuration ---
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client.get_database("face_shape_db")  # You can name your database
analyses_collection = db.get_collection("analyses") # You can name your collection

def upload_image_to_cloudinary(image_file):
    """Uploads an image file to Cloudinary and returns the secure URL."""
    try:
        upload_result = cloudinary.uploader.upload(image_file)
        return upload_result.get("secure_url")
    except Exception as e:
        print(f"Error uploading to Cloudinary: {e}")
        return None

def save_analysis_to_db(image_url, face_shape, measurements):
    """Saves the analysis results to MongoDB."""
    try:
        analysis_data = {
            "image_url": image_url,
            "face_shape": face_shape,
            "measurements": measurements,
            "created_at": datetime.utcnow()
        }
        result = analyses_collection.insert_one(analysis_data)
        return str(result.inserted_id)
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")
        return None 