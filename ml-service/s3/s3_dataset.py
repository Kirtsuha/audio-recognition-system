
import random
from torch.utils.data import Dataset
import librosa

from s3.s3_loader import download_song
from pipeline.dataset import random_segment
from pipeline.augment import add_noise

from config import SR

class S3TripletDataset(Dataset):

    def __init__(self, bucket, keys):
        self.bucket = bucket
        self.keys = keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):

        # anchor / positive
        key_a = self.keys[idx]
        path_a = download_song(self.bucket, key_a)

        audio_a, _ = librosa.load(path_a, sr=SR, mono=True)

        seg_a1 = random_segment(audio_a)
        seg_a2 = random_segment(audio_a)

        # augmentation
        seg_a2 = add_noise(seg_a2)

        # negative
        key_b = random.choice(self.keys)
        path_b = download_song(self.bucket, key_b)

        audio_b, _ = librosa.load(path_b, sr=SR, mono=True)
        seg_b = random_segment(audio_b)

        return seg_a1, seg_a2, seg_b