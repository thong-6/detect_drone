import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

# --- CẤU HÌNH ---
# Đường dẫn đến folder dataset vừa tạo (New_Merged_Dataset)
DATASET_PATH = r"D:\data_train_drone\dataset_after_handle" 

# Định nghĩa Class Names theo đúng thứ tự ID 0-3
CLASS_NAMES = {
    0: 'Airplane',
    1: 'Bird',
    2: 'Drone',
    3: 'Helicopter'
}

def analyze_yolo_dataset(root_path):
    # Dictionary để lưu thống kê
    # Cấu trúc: stats[class_id] = {'bbox_count': 0, 'image_count': 0}
    class_stats = defaultdict(lambda: {'bbox_count': 0, 'image_count': 0})
    
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    total_images = 0
    
    # Duyệt qua các tập train, val, test
    for split in ['train', 'val', 'test']:
        labels_path = os.path.join(root_path, split, 'labels')
        
        if not os.path.exists(labels_path):
            print(f"⚠️ Cảnh báo: Không tìm thấy folder {split} tại {labels_path}")
            continue
            
        # Lấy tất cả file .txt
        txt_files = glob.glob(os.path.join(labels_path, "*.txt"))
        num_files = len(txt_files)
        split_counts[split] = num_files
        total_images += num_files
        
        # Đọc từng file label
        for txt_file in txt_files:
            with open(txt_file, 'r') as f:
                lines = f.readlines()
            
            # Set để theo dõi xem ảnh này ĐÃ tính cho class đó chưa 
            # (Một ảnh có 2 con Drone thì chỉ tính là 1 ảnh chứa Drone)
            classes_in_this_image = set()
            
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                
                class_id = int(parts[0])
                
                # Tăng số lượng BBox
                class_stats[class_id]['bbox_count'] += 1
                classes_in_this_image.add(class_id)
            
            # Tăng số lượng ảnh chứa class tương ứng
            for cid in classes_in_this_image:
                class_stats[cid]['image_count'] += 1

    # --- HIỂN THỊ KẾT QUẢ ---
    print("="*40)
    print(f"TỔNG QUAN DATASET: {root_path}")
    print(f"Tổng số ảnh: {total_images}")
    print(f" - Train: {split_counts['train']}")
    print(f" - Val:   {split_counts['val']}")
    print(f" - Test:  {split_counts['test']}")
    print("="*40)
    print(f"{'Class Name':<15} | {'ID':<3} | {'Số lượng BBox':<15} | {'Số ảnh chứa class':<15}")
    print("-" * 60)
    
    data_for_plot = []
    
    for class_id in sorted(CLASS_NAMES.keys()):
        name = CLASS_NAMES[class_id]
        bboxes = class_stats[class_id]['bbox_count']
        imgs = class_stats[class_id]['image_count']
        print(f"{name:<15} | {class_id:<3} | {bboxes:<15} | {imgs:<15}")
        
        data_for_plot.append({
            'Class': name,
            'BBoxes': bboxes,
            'Images': imgs
        })
    print("="*40)

    # --- VẼ BIỂU ĐỒ ---
    if total_images > 0:
        df = pd.DataFrame(data_for_plot)
        
        plt.figure(figsize=(12, 5))
        
        # Biểu đồ 1: Số lượng BBox
        plt.subplot(1, 2, 1)
        sns.barplot(data=df, x='Class', y='BBoxes', hue='Class', palette='viridis', legend=False)
        plt.title('Số lượng Bounding Boxes (Objects) mỗi Class')
        plt.ylabel('Số lượng Box')
        for i, v in enumerate(df['BBoxes']):
            plt.text(i, v, str(v), ha='center', va='bottom')
            
        # Biểu đồ 2: Số lượng ảnh
        plt.subplot(1, 2, 2)
        sns.barplot(data=df, x='Class', y='Images', hue='Class', palette='magma', legend=False)
        plt.title('Số lượng Ảnh chứa mỗi Class')
        plt.ylabel('Số lượng Ảnh')
        for i, v in enumerate(df['Images']):
            plt.text(i, v, str(v), ha='center', va='bottom')
            
        plt.tight_layout()
        plt.show()
    else:
        print("Không có dữ liệu để vẽ biểu đồ.")

if __name__ == "__main__":
    # Nhớ sửa đường dẫn này trước khi chạy
    analyze_yolo_dataset(DATASET_PATH)