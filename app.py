import os
from flask import Flask, render_template, request, send_from_directory, Response
import cv2
from ultralytics import YOLO
from moviepy.editor import VideoFileClip, AudioFileClip
import numpy as np
from tensorflow import keras
import librosa
import shutil
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static'

# Tạo thư mục nếu chưa có
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
CLASS_LABELS = ['airplane', 'bird', 'drone', 'helicopter']
MODEL_PATH = 'models/sound_classification_model.h5'
audio_model = keras.models.load_model(MODEL_PATH)
# 1. LOAD MODEL YOLO (Visual Model)
# Tự động tải yolov8n.pt nếu chưa có
yolo_model = YOLO('models/best.pt') 
# 1. Định nghĩa hàm trích xuất đặc trưng
def extract_features(file_path, max_pad_len=174, n_mfcc=40):
    """
    Trích xuất đặc trưng MFCC từ file audio.
    """
    try:
        # Load file audio
        audio, sample_rate = librosa.load(file_path, sr=None)
        
        # Trích xuất MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        
        # Pad/Cắt để đồng nhất kích thước
        if mfccs.shape[1] > max_pad_len:
            mfccs = mfccs[:, :max_pad_len]
        else:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# PHẦN MỚI THÊM CHO LIVE CAMERA
# ==========================================

# 1. Hàm tạo luồng frame (Generator Function)
def generate_frames():
    # Mở camera (số 0 thường là webcam mặc định của laptop)
    camera = cv2.VideoCapture('V_AIRPLANE_007.mp4') 
    
    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Đọc frame từ camera
        success, frame = camera.read()
        if not success:
            break
        
        # --- XỬ LÝ YOLO TẠI ĐÂY ---
        # (Giống hệt cách xử lý ảnh tĩnh)
        results = yolo_model(frame, conf = 0.35, iou = 0.5, stream=True)
        
         # 2. Vẽ kết quả lên frame
        for result in results:
            annotated_frame = result.plot()
                
            # 3. Mã hóa ảnh sang định dạng JPEG để gửi qua web
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
        

        # Yield (trả về liên tục) frame theo định dạng multipart/x-mixed-replace
        # Đây là chuẩn để trình duyệt hiểu là luồng video MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    camera.release()

# 2. LOAD MODEL AUDIO (Audio Model)
def predict_audio(audio_path):
    if not audio_path:
        return None, None
    # Ví dụ: Model audio của bạn xử lý file .mp3/.wav và trả về string
    feature = extract_features(audio_path)
    if feature is None:
        return "Error: Can extract features"
    feature_reshaped = feature.reshape(1, feature.shape[0], feature.shape[1], 1)
    prediction = audio_model.predict(feature_reshaped, verbose=0)[0]
    # 4. Lấy kết quả tốt nhất (Top 1)
    best_idx = np.argmax(prediction) # Lấy vị trí có xác suất cao nhất
    best_label = CLASS_LABELS[best_idx]    # Map sang tên
    confidence = prediction[best_idx] # Lấy độ tin cậy (vd: 0.95)
    return best_label, confidence

@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'

        # Lưu file input
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Xử lý dựa trên loại file
        output_filename = 'processed_' + file.filename
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        detected_label =""
        if file_ext in ['.jpg', '.jpeg', '.png']:
            process_image(filepath, output_path)
            media_type = 'image'
        elif file_ext in ['.mp4', '.avi', '.mov']:
            process_video(filepath, output_path)
            media_type = 'video'
        elif file_ext in['.wav', '.mp3']:
            detected_label = process_audio_only(filepath, output_path)
            media_type = 'audio'
        else:
            return "File format not supported"

        return render_template('index.html', result=output_filename, type=media_type, label = detected_label)

    return render_template('index.html', result=None)


# 2. Route để phục vụ luồng video
@app.route('/video_feed')
def video_feed():
    # Trả về Response với mimetype đặc biệt này
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 3. Route cho trang giao diện Live
@app.route('/live')
def live_page():
    # Tạo thêm file templates/live.html (xem bước 2)
    return render_template('live.html')


# --- THÊM HÀM XỬ LÝ RIÊNG CHO AUDIO ---
def process_audio_only(input_path, output_path):
    # 1. Chạy model dự đoán
    label = predict_audio(input_path)
    
    # 2. Copy file gốc sang thư mục static để web có thể phát được
    # (Vì ta không chỉnh sửa nội dung âm thanh, chỉ cần copy qua)
    shutil.copyfile(input_path, output_path)
    
    return label

def process_image(input_path, output_path):
    # Chạy YOLO
    results = yolo_model.predict(source=input_path, conf = 0.35, iou = 0.5, imgsz = 640)
    # Vẽ box và lưu ảnh
    res_plotted = results[0].plot()
    cv2.imwrite(output_path, res_plotted)

def process_video(input_path, output_path):
    # Sử dụng MoviePy để xử lý video và audio dễ dàng hơn
    clip = VideoFileClip(input_path)
    audio_label = ""

    # --- BƯỚC 1: XỬ LÝ AUDIO (Nếu có) ---
    if clip.audio is not None:
        # Trích xuất audio ra file tạm để model audio xử lý
        temp_audio = "temp_audio.wav"
        clip.audio.write_audiofile(temp_audio, logger=None)
        
        # Chạy model Audio
        audio_label, confidence = predict_audio(temp_audio)
        # Xóa file tạm
        if os.path.exists(temp_audio):
            os.remove(temp_audio)   

    # --- BƯỚC 2: XỬ LÝ HÌNH ẢNH + LATE FUSION ---
    # Hàm xử lý từng frame
    def process_frame(frame):
        # Frame từ moviepy là RGB, YOLO nhận tốt
        results = yolo_model(frame, conf = 0.35, iou = 0.5, imgsz=640)
        annotated_frame = results[0].plot() # Vẽ Bounding Box (Visual Output)

        # === LATE FUSION TẠI ĐÂY ===
        # Ghi kết quả Audio lên hình ảnh Video
        if audio_label:
            # Vẽ hình chữ nhật nền cho chữ
            cv2.rectangle(annotated_frame, (10, 10), (400, 60), (0, 0, 255), -1)
            # Viết text kết quả Audio
            cv2.putText(annotated_frame, f"Audio: {audio_label}", (20, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return annotated_frame

    # Áp dụng hàm xử lý cho toàn bộ video
    new_clip = clip.fl_image(process_frame)

    # Giữ lại audio gốc của video (nếu có)
    if clip.audio is not None:
        new_clip = new_clip.set_audio(clip.audio)

    # Xuất file (dùng codec libx264 để web xem được)
    new_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', logger=None)

@app.route('/static/<filename>')
def serve_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)