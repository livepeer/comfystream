from faster_whisper import WhisperModel
import numpy as np
import soundfile as sf

def main():
    # Path to your test audio file (must be a waveform file, e.g., .wav)
    test_audio_path = "test_audio.mp3"  # Replace with the path to your audio file

    # Load the audio file as a waveform
    audio, sample_rate = sf.read(test_audio_path)
    if len(audio.shape) > 1:  # Convert stereo to mono if necessary
        audio = np.mean(audio, axis=1)

    # Initialize the Whisper model
    model = WhisperModel("small", device="cuda", compute_type="float16")

    # Perform transcription
    segments, info = model.transcribe(audio, language=None)

    # Combine the transcription results
    transcription = "".join([segment.text for segment in segments])

    # Print the transcription result
    print("Transcription Result:", transcription)
    print("Detected Language:", info.language)

if __name__ == "__main__":
    main()
