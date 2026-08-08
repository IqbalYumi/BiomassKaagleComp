import os
import time
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tqdm import tqdm  # Digunakan untuk tampilan progress bar

from src.preprocessing import prepare_wide_dataframe, BiomassDataset, get_transforms
from src.models import BiomassMultiTaskModel, PhysicsInformedLoss

def calculate_r2_in_gram_scale(y_true_log, y_pred_log):
    """Konversi kembali dari log(1+x) ke gram asli lalu hitung R2."""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.clip(y_pred, a_min=0.0, a_max=None)
    try:
        return r2_score(y_true, y_pred, multioutput='uniform_average')
    except Exception:
        return 0.0

def main():
    print("=== PROSES TRAINING FULL DATASET (SPLIT 80% TRAIN : 20% VAL) ===")

    # 1. KONFIGURASI PARAMETER & PATH
    EPOCHS = 15
    PATIENCE = 4
    BATCH_SIZE = 8          # Sesuaikan dengan RAM/VRAM Anda
    LEARNING_RATE = 5e-4
    IMG_SIZE = 160
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(BASE_DIR, "data", "csiro-biomass", "train.csv")
    IMG_DIR = os.path.join(BASE_DIR, "data", "csiro-biomass")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. PERSIAPAN DATASET UTAMA
    if not os.path.exists(CSV_PATH):
        print(f"Error: File train.csv tidak ditemukan di {CSV_PATH}")
        return

    # Memuat seluruh data tanpa ada potongan sample
    full_df = prepare_wide_dataframe(CSV_PATH)
    total_images = len(full_df)
    print(f"Total gambar unik yang ditemukan: {total_images} gambar.")

    # SPLIT 80% TRAIN : 20% VALIDATION
    train_df, val_df = train_test_split(full_df, test_size=0.20, random_state=42)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(f"-> Jumlah Data Train (80%): {len(train_df)} gambar")
    print(f"-> Jumlah Data Val   (20%): {len(val_df)} gambar")

    # Dataset dengan Log Transformation
    train_dataset = BiomassDataset(df=train_df, img_dir=IMG_DIR, transform=get_transforms(IMG_SIZE, is_train=True), use_log=True)
    val_dataset = BiomassDataset(df=val_df, img_dir=IMG_DIR, transform=get_transforms(IMG_SIZE, is_train=False), use_log=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. INISIALISASI MODEL (DenseNet121 Pretrained)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device yang digunakan: {device}\n")

    model = BiomassMultiTaskModel(model_name='densenet121', pretrained=True).to(device)

    criterion = PhysicsInformedLoss(alpha=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_val_loss = float('inf')
    patience_counter = 0

    # 4. TRAINING LOOP PER EPOCH DENGAN PROGRESS BAR DETAILED
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch:02d}/{EPOCHS:02d} ---")
        
        # === A. TRAINING PHASE ===
        model.train()
        train_running_loss = 0.0
        
        # Membungkus train_loader dengan tqdm untuk membuat progress bar animasi
        train_pbar = tqdm(
            train_loader, 
            desc=f"Training   [Epoch {epoch:02d}]", 
            unit="batch",
            leave=True
        )

        for images, targets in train_pbar:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item() * images.size(0)
            
            # Update teks loss langsung di sebelah progress bar
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_train_loss = train_running_loss / len(train_dataset)

        # === B. VALIDATION PHASE ===
        model.eval()
        val_running_loss = 0.0
        val_preds_log, val_targets_log = [], []

        val_pbar = tqdm(
            val_loader, 
            desc=f"Validating [Epoch {epoch:02d}]", 
            unit="batch",
            leave=False
        )

        with torch.no_grad():
            for images, targets in val_pbar:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)

                val_running_loss += loss.item() * images.size(0)
                val_preds_log.append(outputs.cpu().numpy())
                val_targets_log.append(targets.cpu().numpy())
                
                val_pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})

        epoch_val_loss = val_running_loss / len(val_dataset)
        val_preds_log = np.vstack(val_preds_log)
        val_targets_log = np.vstack(val_targets_log)
        
        # Hitung skor R2 pada skala Gram asli
        epoch_val_r2 = calculate_r2_in_gram_scale(val_targets_log, val_preds_log)

        # === C. CHECKPOINT & SUMMARY PER EPOCH ===
        status = ""
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            save_path = os.path.join(OUTPUT_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            status = " [MODEL DISIMPAN]"
        else:
            patience_counter += 1
            status = f" [Patience {patience_counter}/{PATIENCE}]"

        # Ringkasan Hasil Epoch
        print(f"-> Hasil Epoch {epoch:02d}: Train Loss = {epoch_train_loss:.4f} | Val Loss = {epoch_val_loss:.4f} | Val R2 = {epoch_val_r2:.4f}{status}")

        # Early Stopping Check
        if patience_counter >= PATIENCE:
            print(f"\n[EARLY STOPPING] Training dihentikan pada epoch {epoch} karena Val Loss stagnan.")
            break

    print("\n=== TRAINING SELESAI ===")
    print(f"Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()