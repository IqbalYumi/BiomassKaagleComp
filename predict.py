import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms

# 1. Setup Path & Import Model EVA-02
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from models_eva import BiomassEVA02TargetQueryModel

# ==========================================
# 2. Setup Direktori & Device
# ==========================================
DATASET_DIR = r"D:\-\GEMASTIK 8\data\csiro-biomass"
TEST_IMG_DIR = os.path.join(DATASET_DIR, "test")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_biomass_eva02_model.pth")
PRED_CSV_PATH = os.path.join(OUTPUT_DIR, "test_predictions.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Menggunakan Device        : {device}")
print(f"Memuat Checkpoint Model   : {CHECKPOINT_PATH}")
print(f"Folder Target Gambar Test : {TEST_IMG_DIR}\n")

# ==========================================
# 3. Pindai MURNI Semua File Gambar di Folder Test
# ==========================================
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.jfif', '.heic')
found_images = []

if not os.path.exists(TEST_IMG_DIR):
    raise FileNotFoundError(f"❌ Folder '{TEST_IMG_DIR}' tidak ditemukan!")

# Pindai langsung seluruh isi folder test (termasuk subfolder jika ada)
for root, _, files in os.walk(TEST_IMG_DIR):
    for f in files:
        if f.lower().endswith(valid_exts):
            full_path = os.path.join(root, f)
            found_images.append((f, full_path))

print(f"🔍 Ditemukan {len(found_images)} file foto di folder test:")
for idx, (fname, _) in enumerate(found_images, 1):
    print(f"   {idx}. {fname}")

if len(found_images) == 0:
    print(f"\n❌ Tidak ada file gambar yang ditemukan di '{TEST_IMG_DIR}'!")
    print("Pastikan file foto (.jpg / .jpeg / .png) sudah berada langsung di dalam folder tersebut.")
    sys.exit(0)

print("-" * 60 + "\n")

# ==========================================
# 4. Transformasi Evaluasi & Load Model
# ==========================================
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = BiomassEVA02TargetQueryModel(model_name='eva02_tiny_patch14_224', pretrained=False)

if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print("✓ Checkpoint model EVA-02 berhasil dimuat!\n")
else:
    raise FileNotFoundError(f"❌ Checkpoint '{CHECKPOINT_PATH}' tidak ditemukan!")

model.to(device)
model.eval()

# ==========================================
# 5. Loop Inferensi Prediksi Seluruh Foto
# ==========================================
target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
predictions_list = []

print("🚀 Memulai prediksi biomassa seluruh foto real-time...")

with torch.no_grad():
    for fname, actual_path in tqdm(found_images, desc="Predicting"):
        try:
            img_pil = Image.open(actual_path).convert("RGB")
            t_img = test_transform(img_pil).unsqueeze(0).to(device)
            
            output = model(t_img).cpu().numpy()[0]
            pred_clip = np.clip(output, a_min=0.0, a_max=None)
            
            row_dict = {'image_filename': fname}
            for idx, t_col in enumerate(target_cols):
                row_dict[t_col] = round(float(pred_clip[idx]), 2)
                
            predictions_list.append(row_dict)
        except Exception as e:
            print(f"[ERROR] Gagal memproses file '{fname}': {e}")

# ==========================================
# 6. Simpan Hasil Prediksi ke CSV & Cetak
# ==========================================
df_predictions = pd.DataFrame(predictions_list)
df_predictions.to_csv(PRED_CSV_PATH, index=False)

print("\n" + "=" * 85)
print(f"✅ PREDIKSI SELESAI UNTUK TOTAL {len(df_predictions)} FOTO REAL-TIME!")
print("=" * 85)
print(f" Hasil CSV Tersimpan di : '{PRED_CSV_PATH}'\n")

print("📋 TABEL REKAP HASIL PREDIKSI BIOMASSA (DALAM GRAM):")
print("-" * 85)
print(df_predictions.to_string(index=False))
print("=" * 85)