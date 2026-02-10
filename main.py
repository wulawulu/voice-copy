# download the pipeline from Huggingface
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1", 
    token="{huggingface-token}")

# run the pipeline locally on your computer
output = pipeline("assets/audio.wav")

# print the predicted speaker diarization 
for turn, speaker in output.speaker_diarization:
    print(f"{speaker} speaks between t={turn.start:.3f}s and t={turn.end:.3f}s")
