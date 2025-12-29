import os
import shutil
from tqdm import tqdm
import yaml

# --- PHẦN 1: CẤU HÌNH ĐƯỜNG DẪN (BẠN SỬA Ở ĐÂY) ---

# Đường dẫn đến folder gốc của bạn
PATH_FOLDER_A = r"D:\data_train_drone\airplane_bird_helicopter" 
PATH_FOLDER_B = r"D:\data_train_drone\drone" 

# Đường dẫn nơi bạn muốn lưu dataset mới đã gộp
OUTPUT_DIR = r"D:\data_train_drone\dataset_after_handle"

# --- PHẦN 2: ĐỊNH NGHĨA MAPPING (ĐÃ CẬP NHẬT THEO YÊU CẦU CỦA BẠN) ---
# Target IDs MỚI: 0: Airplane, 1: Bird, 2: Drone, 3: Helicopter

MAPPING_CONFIG = {
    'Folder_A': {
        'path': PATH_FOLDER_A,
        # Map: ID cũ -> ID mới
        'map': {
            0: 0,  # Airplane -> Airplane (0)
            1: 1,  # Bird -> Bird (1)
            2: 1,  # bird -> Bird (1)
            3: 3   # helicopter -> Helicopter (3)
        }
    },
    'Folder_B': {
        'path': PATH_FOLDER_B,
        'map': {
            0: 2   # drones -> Drone (2)
        }
    }
}

# Các tập con cần xử lý (Lưu ý: Folder của bạn tên là 'valid')
SUBSETS = ['train', 'test', 'valid']

# ---------------------------------------------------------

def create_dir_structure(base_path):
    """Tạo cấu trúc thư mục YOLO chuẩn tại đích."""
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    for subset in SUBSETS:
        # Chuẩn hóa tên folder đích: YOLO dùng 'val' thay vì 'valid'
        target_subset = 'val' if subset == 'valid' else subset 
        
        os.makedirs(os.path.join(base_path, target_subset, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_path, target_subset, 'labels'), exist_ok=True)

def process_dataset():
    create_dir_structure(OUTPUT_DIR)
    
    total_files_copied = 0
    
    for source_name, config in MAPPING_CONFIG.items():
        src_root = config['path']
        mapping = config['map']
        
        print(f"--- Đang xử lý: {source_name} ---")
        
        for subset in SUBSETS:
            # Đường dẫn ảnh và label nguồn
            src_img_dir = os.path.join(src_root, subset, 'images')
            src_lbl_dir = os.path.join(src_root, subset, 'labels')
            
            # Đường dẫn đích (chuyển valid -> val)
            target_subset_name = 'val' if subset == 'valid' else subset
            dst_img_dir = os.path.join(OUTPUT_DIR, target_subset_name, 'images')
            dst_lbl_dir = os.path.join(OUTPUT_DIR, target_subset_name, 'labels')
            
            if not os.path.exists(src_lbl_dir):
                print(f"Bỏ qua {subset} trong {source_name} (Không tìm thấy thư mục labels)")
                continue

            # Lấy danh sách file labels
            label_files = [f for f in os.listdir(src_lbl_dir) if f.endswith('.txt')]
            
            for lbl_file in tqdm(label_files, desc=f"{source_name} - {subset}"):
                src_lbl_path = os.path.join(src_lbl_dir, lbl_file)
                
                # Đọc và xử lý nội dung label
                new_lines = []
                with open(src_lbl_path, 'r') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    
                    class_id = int(parts[0])
                    coords = parts[1:]
                    
                    # Kiểm tra và map ID
                    if class_id in mapping:
                        new_id = mapping[class_id]
                        new_line = f"{new_id} {' '.join(coords)}\n"
                        new_lines.append(new_line)
                
                # CHỈ LƯU nếu file có chứa object hợp lệ
                if new_lines:
                    # 1. Ghi file label mới (thêm tiền tố tên nguồn để tránh trùng)
                    new_filename = f"{source_name}_{lbl_file}"
                    dst_lbl_path = os.path.join(dst_lbl_dir, new_filename)
                    
                    with open(dst_lbl_path, 'w') as f_out:
                        f_out.writelines(new_lines)
                    
                    # 2. Copy file ảnh tương ứng
                    image_found = False
                    base_name = os.path.splitext(lbl_file)[0]
                    # Tìm ảnh với các đuôi phổ biến
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiif', '.webp']:
                        src_img_name = base_name + ext
                        src_img_path = os.path.join(src_img_dir, src_img_name)
                        if os.path.exists(src_img_path):
                            dst_img_name = f"{source_name}_{src_img_name}"
                            shutil.copy2(src_img_path, os.path.join(dst_img_dir, dst_img_name))
                            image_found = True
                            break
                    
                    if image_found:
                        total_files_copied += 1

    print(f"\n✅ Hoàn tất! Đã gộp và xử lý {total_files_copied} ảnh.")
    print(f"Dữ liệu mới nằm tại: {OUTPUT_DIR}")

    # Tạo file data.yaml mới
    create_yaml_file()

def create_yaml_file():
    # Cấu trúc file data.yaml chuẩn YOLOv8
    yaml_content = {
        'path': OUTPUT_DIR,  # Đường dẫn tuyệt đối tới dataset
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 4,
        'names': ['Airplane', 'Bird', 'Drone', 'Helicopter'] # Đã cập nhật thứ tự tên
    }
    
    yaml_path = os.path.join(OUTPUT_DIR, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    print("✅ Đã tạo file data.yaml mới với class names: ['Airplane', 'Bird', 'Drone', 'Helicopter']")

if __name__ == "__main__":
    process_dataset()