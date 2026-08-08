import os
import torch
import torch.nn as nn
import timm

class BiomassMultiTaskModel(nn.Module):
    """
    Model Multi-Task Learning CNN berbasis DenseNet121 dengan teknik 1x1 Convolution 
    untuk kompresi channel dan ekstraksi fitur lintas channel secara efisien.
    """
    def __init__(self, model_name='densenet121', pretrained=True, num_targets=5):
        super(BiomassMultiTaskModel, self).__init__()
        
        # 1. Memuat backbone DenseNet121 bawaan (mengeluarkan feature map 4D)
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            features_only=True  # Mengambil feature map 4D [N, C, H, W]
        )
        
        # 2. Layer 1x1 Convolution (Pointwise Conv)
        # Menekan/mengompres jumlah channel dari DenseNet menjadi 256 channel saja
        self.conv1x1 = nn.Sequential(
            nn.LazyConv2d(out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.SiLU(),  # Activation function
            nn.Dropout2d(p=0.2)
        )
        
        # 3. Global Average Pooling untuk mengubah 4D tensor menjadi 2D vector
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 4. Prediction Head untuk 5 target biomassa
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_targets)
        )

    def forward(self, x):
        # Ekstrak feature map dari backbone (ambil layer terakhir dari DenseNet)
        features_list = self.backbone(x)
        features = features_list[-1]  # Tensor 4D: [Batch, Channel, Height, Width]
        
        # Terapkan 1x1 Convolution
        compressed_features = self.conv1x1(features)
        
        # Global Pooling & Flatten
        pooled_features = self.global_pool(compressed_features)
        
        # Regresi ke 5 target biomassa
        outputs = self.head(pooled_features)
        return outputs

class PhysicsInformedLoss(nn.Module):
    """
    Custom Loss Function yang menambahkan penalti jika
    Dry_Total_g menyimpang dari penjumlahan komponennya.
    Target urutan: [Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g]
    """
    def __init__(self, alpha=0.1):
        super(PhysicsInformedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, preds, targets):
        base_loss = self.mse(preds, targets)
        
        # Penalti konsistensi total: Total ~= Green + Dead + Clover
        pred_components_sum = preds[:, 0] + preds[:, 1] + preds[:, 2]
        pred_total = preds[:, 4]
        
        constraint_loss = self.mse(pred_components_sum, pred_total)
        
        return base_loss + (self.alpha * constraint_loss)

# --- BLOK TEST DRIVE MODEL DENSENET121 ---
if __name__ == "__main__":
    print("=== MENGUJI ARSITEKTUR MODEL DENSENET121 + 1X1 CONVOLUTION ===")
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("1. Inisialisasi model DenseNet121...")
    model = BiomassMultiTaskModel(model_name='densenet121', pretrained=False)
    model.eval()
    
    dummy_images = torch.randn(8, 3, 160, 160)
    dummy_targets = torch.randn(8, 5)
    
    print(f"2. Shape Input Tensor  : {dummy_images.shape}")
    
    with torch.no_grad():
        outputs = model(dummy_images)
        
    print(f"3. Shape Output Tensor : {outputs.shape}")
    
    criterion = PhysicsInformedLoss(alpha=0.1)
    loss = criterion(outputs, dummy_targets)
    print(f"4. Uji Coba Loss Value  : {loss.item():.4f}")
    
    print("\n=== PENGUJIAN DENSENET121 BERHASIL & SIAP DIPAKAI ===")