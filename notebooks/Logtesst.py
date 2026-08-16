import os
import torch

# 1. Tentukan path dinamis berdasarkan lokasi file Logtesst.py ini berada
NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))  # d:\-\GEMASTIK 8\notebooks
BASE_DIR = os.path.abspath(os.path.join(NOTEBOOK_DIR, '..'))  # d:\-\GEMASTIK 8
CHECKPOINT_PATH = os.path.join(
    BASE_DIR, 'outputs', 'best_biomass_eva02_model.pth'
)

print(f"🔍 Memeriksa file checkpoint di: '{CHECKPOINT_PATH}'\n")

# 2. Cek keberadaan file
if not os.path.exists(CHECKPOINT_PATH):
  raise FileNotFoundError(
      f"❌ File '{CHECKPOINT_PATH}' tidak ditemukan. Pastikan nama file di folder outputs/ sudah sesuai!"
  )

# 3. Load Checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')

if isinstance(checkpoint, dict):
  print("📋 Key/Metadata yang tersimpan di dalam file .pth:")
  for key in checkpoint.keys():
    print(f" - {key}")

  if 'history' in checkpoint:
    print('\n✓ Data history training ditemukan di dalam file checkpoint!')
  elif 'best_r2' in checkpoint:
    print(f"\n✓ Rekor Best R² tersimpan: {checkpoint['best_r2']}")
else:
  print("ℹ️ File .pth murni berisi state_dict (hanya matriks bobot model).")