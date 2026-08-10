import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

# Tambahkan folder 'src' ke Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Mengimpor arsitektur model dari src/models_eva.py
from models_eva import BiomassEVA02TargetQueryModel


# ==========================================
# 1. Custom Dataset Khusus Test Data (Auto-Extension)
# ==========================================
class BiomassTestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, test_img_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.test_img_dir = test_img_dir
        self.dataset_root = os.path.dirname(test_img_dir)
        self.transform = transform
        
        # Deteksi otomatis kolom gambar
        self.img_col = None
        for col in ['image_path', 'image_id', 'filename', 'file_name', 'id', 'Image_ID']:
            if col in self.df.columns:
                self.img_col = col
                break
        if self.img_col is None:
            self.img_col = self.df.columns[0]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        raw_img_path = str(row[self.img_col])
        filename = os.path.basename(raw_img_path)
        
        # Bersihkan jika mengandung suffix sample_id (misal: ID1001187975__Dry_Clover_g -> ID1001187975)
        if '__' in filename:
            filename = filename.split('__')[0]
            
        base_paths = [
            os.path.join(self.test_img_dir, filename),
            os.path.join(self.test_img_dir, raw_img_path),
            os.path.join(self.dataset_root, raw_img_path),
            os.path.join(self.dataset_root, filename),
            raw_img_path
        ]
        
        # Coba berbagai ekstensi gambar jika tidak ada ekstensi di string CSV
        extensions = ['', '.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        actual_img_path = None
        for base in base_paths:
            for ext in extensions:
                candidate = base + ext if not base.lower().endswith(('.jpg', '.jpeg', '.png')) else base
                if os.path.exists(candidate):
                    actual_img_path = candidate
                    break
            if actual_img_path:
                break
                
        if actual_img_path is None:
            raise FileNotFoundError(
                f"File gambar test '{filename}' tidak ditemukan (sudah dicoba ekstensi .jpg, .png, .jpeg).\n"
                f"Lokasi pencarian utama: {self.test_img_dir}"
            )
            
        image = Image.open(actual_img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, raw_img_path


# ==========================================
# 2. Pipeline Inference Utama
# ==========================================
def run_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Menggunakan Device untuk Inference: {device}")
    
    # Path Data & Checkpoint Model
    DATASET_DIR = r"D:\-\GEMASTIK 8\data\csiro-biomass"
    test_csv_path = os.path.join(DATASET_DIR, 'test.csv')
    test_img_dir = os.path.join(DATASET_DIR, 'test')  # Folder gambar test
    model_weights_path = r"D:\-\GEMASTIK 8\outputs\best_biomass_eva02_model.pth"
    
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"File test.csv tidak ditemukan di: {test_csv_path}")
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"File model tidak ditemukan di: {model_weights_path}")
    if not os.path.exists(test_img_dir):
        raise FileNotFoundError(f"Folder gambar test tidak ditemukan di: {test_img_dir}")
        
    print(f"Membaca test.csv dari : {test_csv_path}")
    print(f"Membaca Folder Test dari: {test_img_dir}")
    print(f"Memuat Model dari      : {model_weights_path}")
    
    df_test = pd.read_csv(test_csv_path)
    
    # Deteksi gambar unik dari test.csv
    if 'sample_id' in df_test.columns:
        temp_img_paths = df_test['sample_id'].apply(lambda x: str(x).rsplit('__', 1)[0] if '__' in str(x) else str(x))
        unique_images_df = pd.DataFrame({'image_path': temp_img_paths.unique()})
    elif 'image_path' in df_test.columns:
        unique_images_df = df_test[['image_path']].drop_duplicates().reset_index(drop=True)
    else:
        unique_images_df = df_test
        
    # Transforms
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = BiomassTestDataset(unique_images_df, test_img_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Load Model EVA-02
    model = BiomassEVA02TargetQueryModel(model_name='eva02_tiny_patch14_224', pretrained=False)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    model.eval()
    
    target_names = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    results = []
    
    print("\n--- Memulai Prediksi Data Test ---")
    with torch.no_grad():
        for images, img_paths in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            outputs = model(images).cpu().numpy()
            
            # Clip agar nilai biomassa tidak negatif
            outputs = np.clip(outputs, a_min=0.0, a_max=None)
            
            for path, pred in zip(img_paths, outputs):
                for target_idx, target_name in enumerate(target_names):
                    results.append({
                        'image_path': path,
                        'target_name': target_name,
                        'target_value': pred[target_idx]
                    })
                    
    results_df = pd.DataFrame(results)
    
    # ==========================================
    # 3. Matching & Output BEBAS NaN
    # ==========================================
    if 'sample_id' in df_test.columns:
        def parse_key(val):
            val_str = str(val)
            if '__' in val_str:
                img_part, tgt = val_str.rsplit('__', 1)
            else:
                img_part, tgt = val_str, ''
            fname = os.path.basename(img_part)
            fname_no_ext = os.path.splitext(fname)[0]
            return fname_no_ext, tgt

        parsed_test = df_test['sample_id'].apply(parse_key)
        df_test['_key_no_ext'] = [p[0] for p in parsed_test]
        df_test['_tgt_name'] = [p[1] for p in parsed_test]

        results_df['_key_no_ext'] = results_df['image_path'].apply(lambda r: os.path.splitext(os.path.basename(str(r)))[0])

        merged = df_test.merge(
            results_df,
            left_on=['_key_no_ext', '_tgt_name'],
            right_on=['_key_no_ext', 'target_name'],
            how='left'
        )

        merged['target_value'] = merged['target_value'].fillna(0.0)
        target_col_name = 'target' if 'target' in df_test.columns else 'target_value'
        
        final_df = pd.DataFrame({
            'sample_id': df_test['sample_id'],
            target_col_name: merged['target_value']
        })
    else:
        final_df = results_df[['image_path', 'target_name', 'target_value']].fillna(0.0)

    output_dir = r"D:\-\GEMASTIK 8\outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "submission.csv")
    
    final_df.to_csv(output_path, index=False)
    print(f"\n✓ Inference Selesai & BEBAS NaN! File tersimpan di: '{output_path}'")
    print(f"  Jumlah baris terprediksi: {len(final_df)}")
    print("\n--- 5 Baris Teratas Hasil Submission ---")
    print(final_df.head())


if __name__ == "__main__":
    run_inference()