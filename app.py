import cv2
import numpy as np
import pytesseract
from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import ImageConstantROI  # Assuming this is a module with your ROI definitions

app = Flask(__name__)

# Directory to save uploaded files
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
SAMPLE_FOLDER = 'sample'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'pdf', 'png'}

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['SAMPLE_FOLDER'] = SAMPLE_FOLDER

# Create the directories if they don't exist
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    """
    Preprocess the image (e.g., convert to grayscale, blur, and normalize).
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert the image to grayscale if it's not already
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.multiply(image, 1.5)
    # Blur to remove noise
    blured1 = cv2.medianBlur(grayscale_image, 3)
    blured2 = cv2.medianBlur(grayscale_image, 81)
    divided = np.ma.divide(blured1, blured2).data

    # Normalize the image
    normed = np.uint8(255 * divided / divided.max())

    return normed

def cropImageRoi(image, roi):
    roi_cropped = image[
        int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])
    ]
    return roi_cropped

def extractDataFromIdCard(img):
    # Extract all interest data from image
    MODEL_CONFIG = '-l vie --oem 1 --psm 6'
    final = {}
    for key, roi in ImageConstantROI.CCCD.ROIS.items():
        data = ''
        for r in roi:
            crop_img = cropImageRoi(img, r)

            # For a small pixel image only has number, do not preprocess it is better
            if key != 'date_expire':
                crop_img = preprocess_image(crop_img)
            
            data += pytesseract.image_to_string(crop_img, config=MODEL_CONFIG) + ' '

            if data.strip() == '':
                crop_img = preprocess_image(crop_img)
                data += pytesseract.image_to_string(crop_img, config=MODEL_CONFIG) + ' '

        final[key] = data.strip()

    return final

@app.route('/')
def index():
    return render_template('chuyendoi.html')

@app.route('/chuyendoi.html')
def chuyendoi():
    return render_template('chuyendoi.html')

@app.route('/api.html')
def api():
    return render_template('api.html')

@app.route('/login.html')
def login():
    return render_template('login.html')

@app.route('/register.html')
def register():
    return render_template('register.html')

@app.route("/upload", methods=["POST", "GET"])
def main():
    if request.method == 'POST':
        try:
            if 'image' not in request.files:
                return jsonify({"error": "No file part"}), 400

            image = request.files['image']
            if image.filename == '':
                return jsonify({"error": "No selected file"}), 400

            if allowed_file(image.filename):
                image_name = image.filename
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)

                # Read the uploaded image
                file_stream = image.stream
                file_bytes = np.frombuffer(file_stream.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if img is None:
                    return jsonify({"error": "Image could not be decoded"}), 400

                # Resize the image to a fixed size (e.g., 800x800 pixels)
                fixed_size = (800, 800)
                resized_image = cv2.resize(img, fixed_size, interpolation=cv2.INTER_AREA)

                # Get the path of the sample image
                sample_image_path = os.path.join(app.config['SAMPLE_FOLDER'], 'image.png')
                
                # Read the sample image
                sample_image = cv2.imread(sample_image_path, cv2.IMREAD_COLOR)
                if sample_image is None:
                    return jsonify({"error": "Sample image could not be loaded"}), 400

                # Resize both images to a fixed size (800x800 pixels)
                sample_image = cv2.resize(sample_image, fixed_size, interpolation=cv2.INTER_AREA)

                # Initialize the ORB detector
                orb = cv2.ORB_create(nfeatures=4000)

                # Detect keypoints and descriptors
                keypoints1, descriptors1 = orb.detectAndCompute(resized_image, None)
                keypoints2, descriptors2 = orb.detectAndCompute(sample_image, None)

                if descriptors1 is None or descriptors2 is None:
                    return jsonify({"error": "No descriptors found in one or both images"}), 400

                # Match descriptors
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(descriptors1, descriptors2)
                matches = sorted(matches, key=lambda x: x.distance)

                if len(matches) < 4:
                    return jsonify({"error": "Not enough matches found between the images"}), 400

                # Extract the locations of the good matches
                srcPoints = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dstPoints = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                # Find the homography matrix
                matrix_relationship, mask = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
                if matrix_relationship is None:
                    return jsonify({"error": "Homography could not be computed"}), 400

                # Warp the perspective of image1 to match image2
                height, width, channels = sample_image.shape
                aligned_image = cv2.warpPerspective(resized_image, matrix_relationship, (width, height))

                cv2.imwrite(image_path, aligned_image)

                # Preprocess the image
                processed_image = preprocess_image(aligned_image)
                
                # Save the processed image
                processed_image_name = f"processed_{image_name}"
                processed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_image_name)
                cv2.imwrite(processed_image_path, processed_image)

                return jsonify({
                    "response": "File uploaded successfully",
                    "filename": image_name
                }), 200
            else:
                return jsonify({"error": "Unsupported file type"}), 400

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"response": "Send a POST request with an image file"}), 200

@app.route("/uploads/<filename>")
def get_image(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/extracted_data/<filename>")
def get_extracted_data(filename):
    try:
        # Get the path of the uploaded image
        uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Read the uploaded image
        uploaded_image = cv2.imread(uploaded_image_path, cv2.IMREAD_COLOR)
        if uploaded_image is None:
            return jsonify({"error": "Processed image could not be loaded"}), 400

        # Extract data from the ID card image
        extracted_data = extractDataFromIdCard(uploaded_image)

        return jsonify(extracted_data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
