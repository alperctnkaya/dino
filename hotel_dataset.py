import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from urllib.parse import urlparse
from PIL import Image
import pandas as pd
import os

# Default paths for the local filesystem dataset
DEFAULT_METADATA_PATH = "/scratch/plessgrp/hotels/metadata/metadata_4_images_per_hotel.parquet"
DEFAULT_IMAGE_BASE_DIR = "/scratch/plessgrp/hotels"

class HotelDatasetDINO(Dataset):
    """
    Flattened Hotel dataset for DINO Pretraining.
    Returns the multi-crop views for a SINGLE image per __getitem__ call.
    """
    def __init__(self,
                 metadata_path: str = DEFAULT_METADATA_PATH,
                 image_base_dir: str = DEFAULT_IMAGE_BASE_DIR,
                 transform=None):
        super().__init__()
        self.image_base_dir = image_base_dir
        self.transform = transform

        print(f"📂 Loading metadata from {metadata_path}...")
        self.df = pd.read_parquet(metadata_path)
        
        # Pre-compute local image paths from the S3 URLs
        self.df['local_path'] = self.df['path'].apply(self._url_to_local_path)
        
        print(f"   Total usable images: {len(self.df)}")

    def _url_to_local_path(self, url: str) -> str:
        parsed = urlparse(url)
        relative = parsed.path.lstrip('/')
        return os.path.join(self.image_base_dir, relative)

    def __len__(self):
        # Length is simply the number of rows in the metadata
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = row['local_path']
        hotel_id = int(row['hotel_id'])

        try:
            with Image.open(img_path) as f:
                img = f.convert('RGB')   # convert() forces a full read + closes handle cleanly
        except (Image.UnidentifiedImageError, FileNotFoundError, OSError):
            img = Image.new('RGB', (224, 224))

        if self.transform is not None:
            crops = self.transform(img)
            return crops, hotel_id

        return img, hotel_id