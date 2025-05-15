import librosa
import numpy as np

def extract_librosa_features(path: str) -> str:
    y, sr = librosa.load(path, sr=None)

    # Multiple tempo estimates (frame-level)
    tempos = librosa.beat.tempo(y=y, sr=sr, aggregate=None)
    avg_tempo = np.mean(tempos)
    std_tempo = np.std(tempos)

    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1)

    tags = []

    # Tempo level
    if avg_tempo < 60:
        tags.append("very slow tempo")
    elif avg_tempo < 90:
        tags.append("slow tempo")
    elif avg_tempo < 120:
        tags.append("medium tempo")
    elif avg_tempo < 150:
        tags.append("fast tempo")
    else:
        tags.append("very fast tempo")

    # Tempo consistency
    if std_tempo < 3:
        tags.append("highly consistent tempo")
    elif std_tempo < 10:
        tags.append("moderately consistent tempo")
    else:
        tags.append("inconsistent tempo")

    # ZCR as speechiness / noisiness
    if zcr < 0.05:
        tags.append("low noise")
    elif zcr < 0.1:
        tags.append("medium noise")
    else:
        tags.append("high noise")

    # MFCC brightness
    tags.append("bright tone" if mfcc[0] > 0 else "dull tone")

    return ", ".join(tags)
