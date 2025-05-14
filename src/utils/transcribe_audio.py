import whisper
import numpy as np 

_model = whisper.load_model("base")  # use 'tiny' for faster tests

def transcribe_audio(path: str) -> str:
    result = _model.transcribe(path)
    return result["text"]
