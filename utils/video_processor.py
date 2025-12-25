import cv2
import numpy as np
import os
from ultralytics import YOLO
from moviepy.editor import VideoFileClip
import tempfile

class VideoProcessor:
    def __init__(self, model_path='models/best.pt'):
        # Load model YOLO
        self.model = YOLO(model_path)
        
    def process_video(self, input_path, output_path):
        """
        Xử lý video và lưu kết quả dưới dạng MP4
        """
        # Mở video input
        cap = cv2.VideoCapture(input_path)
        
        # Lấy thông số video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Tạo VideoWriter cho MP4 (sử dụng codec H.264)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Chạy YOLO trên frame
            results = self.model(frame)
            
            # Vẽ bounding boxes lên frame
            annotated_frame = results[0].plot()
            
            # Ghi frame đã xử lý
            out.write(annotated_frame)
            
            frame_count += 1
            print(f"Đã xử lý frame {frame_count}")
        
        # Giải phóng tài nguyên
        cap.release()
        out.release()
        
        # Đảm bảo video được encode đúng cách
        self._ensure_mp4_compatibility(output_path)
        
        return output_path
    
    def process_image(self, input_path, output_path):
        """
        Xử lý ảnh
        """
        # Đọc ảnh
        img = cv2.imread(input_path)
        
        # Chạy YOLO
        results = self.model(img)
        
        # Vẽ kết quả
        annotated_img = results[0].plot()
        
        # Lưu ảnh
        cv2.imwrite(output_path, annotated_img)
        
        return output_path
    
    def _ensure_mp4_compatibility(self, video_path):
        """
        Đảm bảo video MP4 có thể phát được trên web
        """
        # Sử dụng moviepy để chuyển đổi nếu cần
        try:
            clip = VideoFileClip(video_path)
            # Tạo file tạm
            temp_output = video_path.replace('.mp4', '_compatible.mp4')
            
            # Xuất với codec phù hợp cho web
            clip.write_videofile(temp_output, codec='libx264', 
                                 audio_codec='aac', 
                                 temp_audiofile='temp-audio.m4a', 
                                 remove_temp=True)
            
            # Thay thế file cũ
            os.replace(temp_output, video_path)
            
        except Exception as e:
            print(f"Không thể tối ưu video: {e}")