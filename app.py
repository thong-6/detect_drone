from flask import Flask, render_template, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
from utils.video_processor import VideoProcessor
import uuid
from datetime import datetime

app = Flask(__name__)

# Cấu hình
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'mkv'}

# Tạo thư mục nếu chưa tồn tại
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Khởi tạo processor
processor = VideoProcessor('models/best.pt')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Không có file được chọn'}), 400
    
    if file and allowed_file(file.filename):
        # Tạo tên file duy nhất
        file_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = secure_filename(f"{timestamp}_{file_id}_{file.filename}")
        
        # Lưu file upload
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        # Xác định loại file và xử lý
        file_ext = filename.rsplit('.', 1)[1].lower()
        result_filename = f"result_{filename.rsplit('.', 1)[0]}.mp4" if file_ext in ['mp4', 'avi', 'mov', 'mkv'] else f"result_{filename}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        
        try:
            if file_ext in ['mp4', 'avi', 'mov', 'mkv']:
                # Xử lý video
                processor.process_video(upload_path, result_path)
                result_type = 'video'
            else:
                # Xử lý ảnh
                processor.process_image(upload_path, result_path)
                result_type = 'image'
            
            return jsonify({
                'success': True,
                'result_url': f'/result/{result_filename}',
                'result_type': result_type,
                'original_name': file.filename
            })
            
        except Exception as e:
            return jsonify({'error': f'Lỗi xử lý: {str(e)}'}), 500
    
    return jsonify({'error': 'Định dạng file không được hỗ trợ'}), 400

@app.route('/result/<filename>')
def get_result(filename):
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename))

@app.route('/list_results')
def list_results():
    results = []
    for file in os.listdir(app.config['RESULT_FOLDER']):
        if file.startswith('result_'):
            file_type = 'video' if file.endswith('.mp4') else 'image'
            results.append({
                'name': file,
                'url': f'/result/{file}',
                'type': file_type,
                'created': os.path.getctime(os.path.join(app.config['RESULT_FOLDER'], file))
            })
    
    # Sắp xếp theo thời gian tạo mới nhất
    results.sort(key=lambda x: x['created'], reverse=True)
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)