import librosa
import torch

def to_mel(audio, sr=16000):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=96
    )

    mel = librosa.power_to_db(mel)
    return torch.tensor(mel).unsqueeze(0)