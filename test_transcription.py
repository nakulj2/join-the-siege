from src.utils.transcribe_audio import transcribe_audio
from src.utils.audio_features import extract_librosa_features

print(transcribe_audio("train_data/podcasts/podcast_3.mp3"))
print(extract_librosa_features("train_data/podcasts/podcast_3.mp3"))