import librosa

# this file contains configuration for generator
paths = {
    "D": "./doum.wav",
    "OTA": "./open_tak.wav",
    "OTI": "./open_tik.wav",
    "PA2": "./pa2.wav",
    "RA": "./ra.wav",
    "T1": "./tik1.wav",
    "T2": "./tik2.wav",
    "S": "./silence.wav",
}


def get_audio_data(symbol, sr=48000):
    path = paths.get(symbol)
    y, _ = librosa.load(path, sr=sr)
    return y
