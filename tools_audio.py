import pyaudio
import librosa
import wave
# ----------------------------------------------------------------------------------------------------------------------
form_1 = pyaudio.paInt16
chans = 1
samp_rate = 44100
chunk = 4096
# ----------------------------------------------------------------------------------------------------------------------
def save_steam_wav(filename_out,frames,audio,chans = 1,form_1=pyaudio.paInt16,samp_rate=44100,):
    with wave.open(filename_out, 'wb') as wavefile:
        width = audio.get_sample_size(form_1)
        wavefile.setnchannels(chans)
        wavefile.setsampwidth(width)
        wavefile.setframerate(samp_rate)
        data= b''.join(frames)
        wavefile.writeframes(data)
        wavefile.close()
    return

# ----------------------------------------------------------------------------------------------------------------------
def trim_file(filename_in,start,stop,filename_out):
    X, fs = librosa.load(filename_in)
    librosa.output.write_wav(filename_out, X[start:stop], fs)
    return
# ----------------------------------------------------------------------------------------------------------------------
def trim_X(X,fs,start,stop,filename_out):
    librosa.output.write_wav(filename_out, X[start:stop], fs)
    return
# ----------------------------------------------------------------------------------------------------------------------