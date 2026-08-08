import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

TARGET_COLS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

def prepare_wide_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    
    if 'image_path' in df.columns:
        df['image_filename'] = df['image_path'].apply(lambda x: os.path.basename(x))
    elif 'sample_id' in df.columns:
        df['image_filename'] = df['sample_id'].apply(lambda x: x.split('__')[0] + '.jpg')
        
    if 'target_name' in df.columns and 'target' in df.columns:
        wide_df = df.pivot(
            index='image_filename', 
            columns='target_name', 
            values='target'
        ).reset_index()
        
        meta_cols = ['image_filename', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']
        available_meta = [c for c in meta_cols if c in df.columns]
        if len(available_meta) > 1:
            meta_df = df[available_meta].drop_duplicates(subset=['image_filename'])
            wide_df = pd.merge(wide_df, meta_df, on='image_filename', how='left')
            
        return wide_df
    return df

class BiomassDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False, use_log=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        self.use_log = use_log

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image_filename']
        
        img_path = os.path.join(self.img_dir, "train_images", img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, "train", img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if not self.is_test:
            raw_targets = row[TARGET_COLS].values.astype(np.float32)
            # Terapkan Log Transform log(1 + x) jika use_log=True
            if self.use_log:
                targets = np.log1p(raw_targets)
            else:
                targets = raw_targets
            return image, torch.tensor(targets, dtype=torch.float32)

        return image, img_name

def get_transforms(img_size=160, is_train=True):
    if is_train:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.05),
            T.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])