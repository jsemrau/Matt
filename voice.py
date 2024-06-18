import os
import sounddevice as sd
import numpy as np
try:
    from nemo_text_processing.text_normalization.normalize import Normalizer
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "The package `nemo_text_processing` was not installed in this environment. Please refer to"
        " https://github.com/NVIDIA/NeMo-text-processing and install this package before using "
        "this script"
    )

# By using XTTS you agree to CPML license https://coqui.ai/cpml
os.environ["COQUI_TOS_AGREED"] = "1"
from TTS.api import TTS
from pydub import AudioSegment
import uuid
import io
from scipy.io.wavfile import write
class SpeakAgent():

    def __init__(self) -> None:

        print(" I am a happy happy Matty")


    def speak(self, text,store_local=True):

        new_uuid = uuid.uuid4()
        text_normalizer = Normalizer(input_case="cased", lang="en")
        normalized_text = text_normalizer.normalize(text, verbose=True, punct_post_process=True)
        print("Downloading if not downloaded Coqui XTTS V2")
        device="cuda:4"
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

        if store_local:

            tts.tts_to_file(normalized_text, speaker_wav="male.wav", language="en", file_path=f"./audio/{new_uuid}.wav")
            audio = AudioSegment.from_wav(f"./audio/{new_uuid}.wav")
            # Export as MP3
            #audio.export(f"{new_uuid}.mp3", format="mp3")
            return f"{new_uuid}.wav"
        else:
            wav_data = tts.tts(normalized_text, speaker_wav="male.wav", language="en")
            #sd.play(wav_data, samplerate=22050)  # Adjust the sample rate as needed
            # sd.wait()  # Wait until the audio is done playing
            #wav_bytes = self.convert_wav_to_bytes(wav_data, 22050)
            return wav_data
        #del tts
        print("XTTS downloaded")


    def convert_wav_to_bytes(self,wav_data, sample_rate):

        if not isinstance(wav_data, np.ndarray):
            wav_data = np.array(wav_data)

        # Convert the WAV data to a bytes-like object
        wav_io = io.BytesIO()
        write(wav_io, sample_rate, wav_data.astype(np.int16))
        wav_io.seek(0)
        return wav_io
#aspek=SpeakAgent()
#aspek.speak("Fuck you!")