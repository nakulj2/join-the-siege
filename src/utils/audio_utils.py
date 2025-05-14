from pydub import AudioSegment

def get_duration(filepath: str) -> float:
    audio = AudioSegment.from_file(filepath)
    return len(audio) / 1000 