import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm
import torchvision.transforms as T

# ==========================================
# 1. Setup Direktori
# ==========================================
DATASET_DIR = r"D:\-\GEMASTIK 8\data\csiro-biomass"
TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
RAW_IMG_DIR = os.path.join(DATASET_DIR, "train_images")

# Folder Output Khusus Augmentation Data
AUG_IMG_DIR = os.path.join(DATASET_DIR, "Augmentation_data")
OUTPUT_CSV = os.path.join(DATASET_DIR, "train_aug.csv")

os.makedirs(AUG_IMG_DIR, exist_ok=True)


# ==========================================
# 2. Fungsi Enhancement Citra Lanjutan
# ==========================================

def apply_clahe(img_pil: Image.Image) -> Image.Image:
    """Menerapkan CLAHE pada kanal Luminance (L) dalam ruang warna LAB."""
    img_np = np.array(img_pil)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Inisialisasi CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    enhanced_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced_np)

def apply_sharpen(img_pil: Image.Image) -> Image.Image:
    """Menajamkan tekstur tepi daun dan serat rumput menggunakan Sharpening Kernel."""
    img_np = np.array(img_pil)
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    sharpened_np = cv2.filter2D(img_np, -1, kernel)
    return Image.fromarray(sharpened_np)

def apply_color_enhance(img_pil: Image.Image) -> Image.Image:
    """Memperjelas saturasi dan kontras warna vegetasi (Hijau vs Cokelat)."""
    # Enhancer Saturasi Warna
    converter_color = ImageEnhance.Color(img_pil)
    img_colored = converter_color.enhance(1.25)
    
    # Enhancer Kontras
    converter_contrast = ImageEnhance.Contrast(img_colored)
    img_enhanced = converter_contrast.enhance(1.15)
    return img_enhanced


# ==========================================
# 3. Pipeline Augmentasi & Enhancements
# ==========================================
# Transformasi Geometris
geo_transforms = {
    "rot90": T.RandomRotation(degrees=(90, 90)),
    "rot180": T.RandomRotation(degrees=(180, 180)),
    "rot270": T.RandomRotation(degrees=(270, 270)),
    "hflip": T.RandomHorizontalFlip(p=1.0),
    "vflip": T.RandomVerticalFlip(p=1.0),
}

print(f"Membaca data train asli dari : {TRAIN_CSV}")
df = pd.read_csv(TRAIN_CSV)

# Deteksi Kolom Gambar
if 'image_path' in df.columns:
    img_col = 'image_path'
else:
    img_col = df.columns[0]

is_long_format = 'target_name' in df.columns or 'variable' in df.columns
if is_long_format:
    unique_imgs = df[img_col].unique()
else:
    unique_imgs = df[img_col].values

print(f"Folder Output Gambar Baru : {AUG_IMG_DIR}")
print(f"Memulai Ekspansi Dataset dari {len(unique_imgs)} gambar asli...\n")

new_rows = []

for img_path in tqdm(unique_imgs, desc="Augmenting & Enhancing"):
    fname = os.path.basename(str(img_path))
    
    candidate_paths = [
        os.path.join(RAW_IMG_DIR, fname),
        os.path.join(RAW_IMG_DIR, str(img_path)),
        os.path.join(DATASET_DIR, str(img_path))
    ]
    actual_path = next((p for p in candidate_paths if os.path.exists(p)), None)
    
    if actual_path is None:
        continue
        
    img_orig = Image.open(actual_path).convert("RGB")
    name_no_ext, ext = os.path.splitext(fname)
    
    # List Pasangan (Nama_Suffix, Image_PIL)
    generated_variants = [
        (fname, img_orig),                                         # 1. Gambar Asli
        (f"{name_no_ext}_clahe{ext}", apply_clahe(img_orig)),       # 2. CLAHE Enhanced
        (f"{name_no_ext}_sharp{ext}", apply_sharpen(img_orig)),     # 3. Sharpened
        (f"{name_no_ext}_color{ext}", apply_color_enhance(img_orig))# 4. Color & Contrast Boost
    ]
    
    # 5-9. Tambahkan Variasi Geometris (Rotasi & Flip) dari Gambar CLAHE/Color
    for tag, trans in geo_transforms.items():
        # Terapkan rotasi/flip pada gambar asli
        aug_img = trans(img_orig)
        generated_variants.append((f"{name_no_ext}_{tag}{ext}", aug_img))
        
    # Kombinasi CLAHE + Rotasi 90
    clahe_img = generated_variants[1][1]
    generated_variants.append((f"{name_no_ext}_clahe_rot90{ext}", geo_transforms["rot90"](clahe_img)))

    # Save semua variasi gambar dan catat di DataFrame CSV
    for new_fname, img_var in generated_variants:
        img_var.save(os.path.join(AUG_IMG_DIR, new_fname))
        
        # Kecuali gambar asli (yang sudah ada di df), tambahkan baris baru untuk variasi lain
        if new_fname != fname:
            if is_long_format:
                sub_df = df[df[img_col] == img_path].copy()
                sub_df[img_col] = new_fname
                if 'sample_id' in sub_df.columns:
                    sub_df['sample_id'] = sub_df['sample_id'].apply(
                        lambda s: f"{new_fname}__{str(s).split('__')[1]}" if '__' in str(s) else f"{new_fname}__{s}"
                    )
                new_rows.append(sub_df)
            else:
                sub_df = df[df[img_col] == img_path].copy()
                sub_df[img_col] = new_fname
                new_rows.append(sub_df)

# Gabungkan Seluruh Data
df_aug_all = pd.concat([df] + new_rows, ignore_index=True)
df_aug_all.to_csv(OUTPUT_CSV, index=False)

print("\n" + "="*60)
print("✓ PROSES AUGMENTASI & ENHANCEMENT SELESAI SUKSES!")
print("="*60)
print(f" Total Sampel Asli  : {len(df)} baris")
print(f" Total Sampel Baru  : {len(df_aug_all)} baris (~9x LIPAT!)")
print(f" File CSV Baru      : '{OUTPUT_CSV}'")
print(f" Folder Gambar Baru : '{AUG_IMG_DIR}'")