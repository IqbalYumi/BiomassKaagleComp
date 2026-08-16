import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

# Tambahkan folder 'src' ke Python path agar dapat mengimpor models_eva.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Mengimpor model dari src/models_eva.py
from models_eva import BiomassEVA02TargetQueryModel


# ==========================================
# 1. Helper: Deteksi Kolom Gambar & Wide Format
# ==========================================
def find_image_col(df: pd.DataFrame) -> str:
    """Otomatis mencari nama kolom gambar pada DataFrame."""
    candidates = ['image_path', 'image_id', 'filename', 'file_name', 'image_name', 'image', 'id', 'Image_ID', 'ID']
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        if col not in ['target_name', 'target_value', 'target', 'value', 'sample_id', 'variable', 'label', 'Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']:
            return col
    raise KeyError(f"Tidak dapat menemukan kolom gambar di CSV. Kolom yang tersedia: {list(df.columns)}")


def ensure_wide_format(df: pd.DataFrame, img_col: str) -> pd.DataFrame:
    """Otomatis mengubah Long Format ke Wide Format secara presisi."""
    target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    
    # Jika 5 kolom target sudah ada, berarti sudah Wide Format
    if all(col in df.columns for col in target_cols):
        return df

    # 1. Cari kolom nama target
    target_name_col = None
    for col in ['target_name', 'variable', 'label', 'category', 'target_type']:
        if col in df.columns:
            target_name_col = col
            break

    # 2. Cari kolom nilai biomassa
    target_val_col = None
    for col in ['target', 'target_value', 'value', 'weight', 'val', 'target_val']:
        if col in df.columns and col != target_name_col:
            target_val_col = col
            break

    if target_val_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in [img_col, target_name_col, 'sample_id', 'id']]
        if numeric_cols:
            target_val_col = numeric_cols[0]

    if target_name_col and target_val_col:
        print(f"Mengubah Long Format ke Wide Format (index='{img_col}', columns='{target_name_col}', values='{target_val_col}')...")
        df_wide = df.pivot(index=img_col, columns=target_name_col, values=target_val_col).reset_index()
        df_wide.columns.name = None
        
        # Pastikan kelima kolom target ada & dikonversi ke float32 secara aman
        for tc in target_cols:
            if tc not in df_wide.columns:
                df_wide[tc] = 0.0
            df_wide[tc] = pd.to_numeric(df_wide[tc], errors='coerce').fillna(0.0).astype(np.float32)
            
        return df_wide
    else:
        raise KeyError(
            f"Gagal memformat data ke Wide Format. Kolom yang ada di train.csv: {list(df.columns)}"
        )


# ==========================================
# 2. Custom Dataset PyTorch
# ==========================================
class BiomassDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.dataset_root = os.path.dirname(img_dir)
        self.transform = transform
        self.target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
        self.img_col = find_image_col(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        raw_img_path = str(row[self.img_col])
        filename = os.path.basename(raw_img_path)
        
        # Lokasi pencarian file gambar
        candidate_paths = [
            os.path.join(self.img_dir, filename),
            os.path.join(self.img_dir, raw_img_path),
            os.path.join(self.dataset_root, raw_img_path),
            os.path.join(self.dataset_root, 'Augmentation_data', filename),
            os.path.join(self.dataset_root, filename),
            raw_img_path
        ]
        
        actual_img_path = None
        for path in candidate_paths:
            if os.path.exists(path):
                actual_img_path = path
                break
                
        if actual_img_path is None:
            raise FileNotFoundError(f"File gambar '{filename}' tidak ditemukan di {self.img_dir}.")
            
        image = Image.open(actual_img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        targets = row[self.target_cols].values.astype(np.float32)
        return image, torch.tensor(targets)


# ==========================================
# 3. Data Transforms
# ==========================================
def get_transforms(img_size=224):
    # Karena dataset offline di folder Augmentation_data sudah memiliki variasi rotasi & CLAHE,
    # kita gunakan transform standar + ringan agar tidak melampaui batas saturasi.
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


# ==========================================
# 4. Pipeline Training Utama
# ==========================================
def train_pipeline(df: pd.DataFrame, img_dir: str, config: dict):
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')
    print(f"Menggunakan Device: {device}")
    
    output_dir = config.get('output_dir', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Model terbaik akan disimpan ke folder: '{output_dir}/'")
    
    # Deteksi Kolom Gambar & Wide Format
    img_col = find_image_col(df)
    print(f"Kolom gambar terdeteksi: '{img_col}'")
    
    df = ensure_wide_format(df, img_col)
    
    # Split Data Train & Validation
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Data Train: {len(train_df)} gambar | Data Validation: {len(val_df)} gambar")
    
    # DataLoader Setup
    train_transform, val_transform = get_transforms(img_size=config['img_size'])
    
    train_dataset = BiomassDataset(train_df, img_dir, transform=train_transform)
    val_dataset = BiomassDataset(val_df, img_dir, transform=val_transform)
    
    num_workers = 2 if is_cuda else 0
    pin_mem = is_cuda
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_mem
    )
    
    # Inisialisasi Arsitektur Model
    model = BiomassEVA02TargetQueryModel(
        model_name=config['backbone_name'],
        pretrained=True,
        drop_rate=config['drop_rate']
    ).to(device)

    # -------------------------------------------------------------
    # 1. PARTIAL FREEZING BACKBONE
    # -------------------------------------------------------------
    freeze_pct = config.get('freeze_backbone_pct', 0.6)
    if freeze_pct > 0:
        backbone_params = list(model.backbone.parameters())
        num_to_freeze = int(len(backbone_params) * freeze_pct)
        for i, param in enumerate(backbone_params):
            if i < num_to_freeze:
                param.requires_grad = False
        print(f"✓ Berhasil mem-freeze {freeze_pct*100:.0f}% layer awal backbone ({num_to_freeze}/{len(backbone_params)} parameter tensor).")

    # -------------------------------------------------------------
    # 2. RESUME TRAINING / CHECKPOINT LOADING
    # -------------------------------------------------------------
    save_path = os.path.join(output_dir, 'best_biomass_eva02_model.pth')
    best_val_r2 = config.get('baseline_r2', -float('inf'))

    if os.path.exists(save_path) and config.get('resume', True):
        print(f"\n[INFO] Memuat checkpoint terbaik sebelumnya dari: '{save_path}'")
        model.load_state_dict(torch.load(save_path, map_location=device))
        if best_val_r2 == -float('inf'):
            best_val_r2 = 0.5886  # Rekor Mean R2 terbaik sebelumnya
        print(f"✓ Bobot dimuat! Baseline Mean R² yang harus dilewati: {best_val_r2:.4f}\n")
    else:
        print("\n[INFO] Memulai pelatihan dari awal (Pretrained ImageNet).\n")
        best_val_r2 = -float('inf')

    # Optimizer, Loss, & Scheduler
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if is_cuda else None

    patience = config['patience']
    patience_counter = 0

    # Cetak Header Tabel Evaluasi
    table_divider = "=" * 97
    sub_divider   = "-" * 97
    
    print("\n" + table_divider)
    print(f"| {'Epoch':^7} | {'Train Loss':^11} | {'Val Loss':^10} | {'Mean R²':^9} | {'Clover':^8} | {'Dead':^8} | {'Green':^8} | {'Total':^8} | {'GDM':^8} |")
    print(table_divider)

    for epoch in range(1, config['epochs'] + 1):
        # --- TRAIN PHASE ---
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{config['epochs']:02d}] - Train", leave=False)
        for images, targets in train_pbar:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
            train_loss += loss.item() * images.size(0)
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        scheduler.step()
        train_loss = train_loss / len(train_dataset)
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        
        val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch:02d}/{config['epochs']:02d}] - Val  ", leave=False)
        with torch.no_grad():
            for images, targets in val_pbar:
                images, targets = images.to(device), targets.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * images.size(0)
                val_preds.append(outputs.cpu().numpy())
                val_targets.append(targets.cpu().numpy())
                
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
        val_loss = val_loss / len(val_dataset)
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        
        # Hitung R2 Keseluruhan dan R2 Per-Target
        r2_per_target = r2_score(val_targets, val_preds, multioutput='raw_values')
        val_r2 = float(np.mean(r2_per_target))
        
        r2_clover, r2_dead, r2_green, r2_total, r2_gdm = r2_per_target
        
        # Cetak Baris Tabel Evaluasi
        print(f"| {epoch:02d}/{config['epochs']:02d}   | {train_loss:^11.4f} | {val_loss:^10.4f} | {val_r2:^9.4f} | {r2_clover:^8.4f} | {r2_dead:^8.4f} | {r2_green:^8.4f} | {r2_total:^8.4f} | {r2_gdm:^8.4f} |")
        
        # --- CHECKPOINT & EARLY STOPPING ---
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"   [★] Model Terbaik Disimpan (Best Mean R² Baru: {best_val_r2:.4f})")
            print(sub_divider)
        else:
            patience_counter += 1
            print(f"   [-] Mean R² tidak membaik ({patience_counter}/{patience})")
            print(sub_divider)
            if patience_counter >= patience:
                print(table_divider)
                print(f"\n[Early Stopping] Training dihentikan di Epoch {epoch} karena Mean R² tidak meningkat selama {patience} epoch.")
                break

    print(table_divider)
    print(f"\nTraining Selesai! Skor Validation Mean R² Terbaik: {best_val_r2:.4f}")


# ==========================================
# 5. Konfigurasi Hyperparameters & Eksekusi
# ==========================================
if __name__ == "__main__":
    hyperparameters = {
        'backbone_name': 'eva02_tiny_patch14_224',
        'img_size': 224,
        'batch_size': 8,             # Batch size 16 untuk dataset 3.570 gambar
        'epochs': 50,
        'patience': 25,               # Early stopping window
        'lr': 5e-5,                   # Learning rate stabil untuk dataset augmented
        'weight_decay': 0.05,         # Weight decay optimal
        'drop_rate': 0.35,            # Dropout rate untuk mencegah overfitting
        'freeze_backbone_pct': 0.6,   # Freeze 60% layer awal backbone
        'resume': True,               # Melanjutkan dari checkpoint best_biomass_eva02_model.pth jika ada
        'baseline_r2': 0.9780,        # Rekor R2 minimal yang harus dilewati
        'output_dir': 'outputs'
    }
    
    DATASET_DIR = r"D:\-\GEMASTIK 8\data\csiro-biomass"
    
    # MENGGUNAKAN DATASET HASIL AUGMENTASI & ENHANCEMENT
    csv_path = os.path.join(DATASET_DIR, 'train_aug.csv')
    img_dir = os.path.join(DATASET_DIR, 'Augmentation_data')
    
    print(f"Membaca CSV dari: {csv_path}")
    print(f"Membaca Folder Gambar dari: {img_dir}")
    
    df_train = pd.read_csv(csv_path)
    train_pipeline(df_train, img_dir, hyperparameters)