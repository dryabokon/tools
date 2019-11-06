import pyaudio
import librosa
import wave
import numpy
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
def merge_audio_files(list_of_files, filename_out):

    res=[]
    sr = samp_rate
    for filename in list_of_files:
        x, sr = librosa.load(filename)
        res.append(x)

    res = numpy.concatenate(res)

    librosa.output.write_wav(filename_out, res, sr)
    return
# ----------------------------------------------------------------------------------------------------------------------