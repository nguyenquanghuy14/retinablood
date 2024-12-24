
from flask import Flask, request, render_template, jsonify
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# Đường dẫn tới mô hình đã huấn luyện
model_path = "path_to_your_model/retina_vessel_segmentation.keras"
model = load_model(model_path)

# Kích thước ảnh đầu vào
IMG_HEIGHT, IMG_WIDTH = 512, 512

# Hàm tiền xử lý ảnh
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# Hàm xử lý ảnh và dự đoán
def segment_image(image_path):
    input_image = preprocess_image(image_path)
    mask_prediction = model.predict(input_image)[0]
    mask_prediction = (mask_prediction > 0.5).astype(np.uint8) * 255
    return mask_prediction

# Trang chính (hiển thị giao diện upload)
@app.route('/')
def home():
    return render_template('index.html')

# API nhận file ảnh và trả về kết quả
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file:
        # Lưu file gốc
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        # Xử lý và dự đoán ảnh
        mask = segment_image(file_path)

        # Lưu ảnh kết quả
        result_path = os.path.join("static", "result.png")
        Image.fromarray(mask.squeeze()).save(result_path)

        # Trả về kết quả
        return jsonify({"result_image": "/static/result.png"})

    return jsonify({"error": "File not processed"}), 500

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)
