import librosa
import numpy as np

def extract_librosa_features(path: str) -> np.ndarray:
    y, sr = librosa.load(path, sr=None)

    tempo = librosa.beat.tempo(y=y, sr=sr)[0]
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1)

    return np.concatenate(([tempo, zcr], mfcc))  # final shape: 1D vector
