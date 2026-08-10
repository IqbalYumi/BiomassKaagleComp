import torch
import torch.nn as nn
import timm

class TargetQueryCrossAttention(nn.Module):
    """
    Modul Cross-Attention untuk mengekstraksi fitur dari 5 target biomassa
    menggunakan 5 Learnable Target Queries di ruang sequence 1D.
    """
    def __init__(self, embed_dim: int, num_targets: int = 5, num_heads: int = 4):
        super().__init__()
        self.num_targets = num_targets
        # 5 Learnable Target Queries (1 vektor per target biomassa)
        self.target_queries = nn.Parameter(torch.randn(1, num_targets, embed_dim) * 0.02)
        
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_k = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True
        )

    def forward(self, patch_tokens: torch.Tensor):
        B = patch_tokens.shape[0]
        queries = self.target_queries.expand(B, -1, -1)
        
        norm_q = self.norm_q(queries)
        norm_k = self.norm_k(patch_tokens)
        
        attn_out, attn_weights = self.cross_attn(
            query=norm_q,
            key=norm_k,
            value=norm_k
        )
        return attn_out, attn_weights


class BiomassEVA02TargetQueryModel(nn.Module):
    """
    Model Arsitektur Murni Vision Transformer (EVA-02) + Target Query Cross-Attention
    untuk Regresi 5 Target Biomassa.
    """
    def __init__(
        self, 
        model_name: str = 'eva02_tiny_patch14_224', 
        num_targets: int = 5, 
        pretrained: bool = True, 
        drop_rate: float = 0.2
    ):
        super().__init__()
        
        # Load backbone EVA-02 tanpa num_classes=0 agar pretrained weights termuat sempurna
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            drop_rate=drop_rate
        )
        
        # Hapus head klasifikasi bawaan ImageNet secara aman
        self.backbone.reset_classifier(0)
        
        embed_dim = self.backbone.num_features
        
        # Cross Attention Module
        self.target_cross_attn = TargetQueryCrossAttention(
            embed_dim=embed_dim,
            num_targets=num_targets,
            num_heads=4
        )
        
        # 5 Regressor MLP Terpisah (1 Head Khusus per Target)
        self.regressors = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Dropout(drop_rate),
                nn.Linear(embed_dim, 128),
                nn.GELU(),
                nn.Dropout(drop_rate / 2),
                nn.Linear(128, 1)
            ) for _ in range(num_targets)
        ])

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        # 1. Ekstraksi sequence of patch tokens 1D (B, N, D)
        tokens = self.backbone.forward_features(x)
        
        # 2. Hapus CLS Token jika ada (N = 257 -> 256 patch tokens)
        if tokens.dim() == 3 and tokens.shape[1] == 257:
            tokens = tokens[:, 1:, :]
            
        # 3. Target Query Cross-Attention
        target_features, attn_weights = self.target_cross_attn(tokens)
        
        # 4. Prediksi 5 target via MLP masing-masing
        preds = []
        for i in range(5):
            tf_i = target_features[:, i, :]
            pred_i = self.regressors[i](tf_i)
            preds.append(pred_i)
            
        out = torch.cat(preds, dim=1)
        
        if return_attn:
            return out, attn_weights
        return out


# ==========================================
# Uji Coba Model (Sanity Check)
# ==========================================
if __name__ == "__main__":
    print("Menguji Arsitektur Model BiomassEVA02TargetQueryModel...")
    
    # Inisialisasi model
    model = BiomassEVA02TargetQueryModel(
        model_name='eva02_tiny_patch14_224', 
        pretrained=True
    )
    
    # Dummy input: Batch=4, RGB Channels=3, Height=224, Width=224
    dummy_img = torch.randn(4, 3, 224, 224)
    
    # Forward pass
    output, weights = model(dummy_img, return_attn=True)
    
    print(f"✓ Output Prediksi Shape : {output.shape}  (Sesuai: [4, 5])")
    print(f"✓ Attention Weight Shape: {weights.shape} (Sesuai: [4, 5, 256])")
    print("✓ Model siap di-import ke train.py atau inference.py!")