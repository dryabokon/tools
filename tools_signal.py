from scipy import fftpack
import numpy
import scipy.interpolate
import pandas as pd
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------------------------------------------------------
import tools_IO
# ---------------------------------------------------------------------------------------------------------------------
def fit_sample_into_range(signal, N):
    interp = scipy.interpolate.interp1d(numpy.arange(signal.shape[0]), signal)
    signal_interp = interp(numpy.linspace(0, signal.shape[0]-1, N))
    return signal_interp
# ---------------------------------------------------------------------------------------------------------------------
def zerro_crossings_down(signal):
    x = signal
    indices = numpy.where((x[1:] >= 0) & (x[:-1] < 0))[0]
    return indices
# ---------------------------------------------------------------------------------------------------------------------
def zerro_crossings_up(signal):
    x = signal
    indices = numpy.where((x[1:] < 0) & (x[:-1] >= 0))[0]
    return indices
# ---------------------------------------------------------------------------------------------------------------------
def get_phases(signal,framestamps,frame_start,frame_stop,num_phases):
    phase = numpy.full(signal.shape[0],-1, dtype=numpy.int)

    signal2= signal-signal[frame_start-framestamps[0]:frame_stop-framestamps[0]].mean()

    idx_down = zerro_crossings_down(signal2)
    idx_up   = zerro_crossings_up  (signal2)

    idx_down = numpy.delete(idx_down, numpy.where(idx_down < frame_start-framestamps[0]))
    idx_down = numpy.delete(idx_down, numpy.where(idx_down > frame_stop-framestamps[0]))

    idx_up = numpy.delete(idx_up, numpy.where(idx_up < frame_start-framestamps[0]))
    idx_up = numpy.delete(idx_up, numpy.where(idx_up > frame_stop-framestamps[0]))


    if idx_up[0]<idx_down[0]:
        idx_up = numpy.delete(idx_up,0)

    for i in numpy.arange(0, idx_down.shape[0] - 1, 1):
        start, stop = idx_down[i], idx_up[i]
        phase[start:stop] = numpy.linspace(0, int(num_phases / 2), stop - start).astype(int)
        start, stop = idx_up[i], idx_down[i+1]
        phase[start:stop] = numpy.linspace(int(num_phases / 2), num_phases-1, stop - start).astype(int)

    return phase
# ---------------------------------------------------------------------------------------------------------------------
def get_average(signal,framestamps,frame_start,frame_stop):
    avg = numpy.full(signal.shape[0], signal[frame_start-framestamps[0]:frame_stop-framestamps[0]].mean(), dtype=numpy.float32)
    return avg
# ---------------------------------------------------------------------------------------------------------------------
def get_best_cycle(signal,approxim,phases):
    num_phases = int(numpy.max(phases)+1)

    idx_start0 =  1 + numpy.where((phases[1:] == 0) & (phases[:-1] == -1))[0]
    idx_start2 = 1 + numpy.where((phases[1:] == 0) & (phases[:-1] == num_phases-1))[0]
    idx_start = numpy.unique(numpy.hstack((idx_start0,idx_start2)))

    idx_stop0 = numpy.where((phases[:-1] == num_phases-1) & (phases[1:] == 0))[0]
    idx_stop2 = numpy.where((phases[:-1] == num_phases - 1) & (phases[1:] == -1))[0]
    idx_stop  = numpy.unique(numpy.hstack((idx_stop0, idx_stop2)))

    #idx_stop = idx_start[1:]-1
    #idx_start = idx_start[:-1]

    q = numpy.full(idx_start.shape[0],1e6)
    for c in range(0, numpy.minimum(idx_start.shape[0],idx_stop.shape[0])):
        start, stop = idx_start[c],idx_stop[c]

        v = signal[start:stop] - approxim[start:stop]
        q[c] = numpy.sum(v**2)/v.shape[0]
    idx = numpy.argmin(q)


    ph1= phases[idx_start[idx]]
    ph2 = phases[idx_stop[idx]]

    return idx_start[idx], idx_stop[idx]
# ---------------------------------------------------------------------------------------------------------------------
def approximate_periodical0(signal,phase,timerange=None):

    N=5
    F = numpy.poly1d(numpy.polyfit(phase[numpy.where(phase != 0)], signal[numpy.where(phase != 0)], N))

    if timerange is None:
        sign_approx = F(phase)
    else:
        sign_approx = F(timerange)
    return sign_approx
# ---------------------------------------------------------------------------------------------------------------------
def approximate_periodical(signal,phase,timerange=None):

    N = numpy.max(phase)-numpy.min(phase)+1
    m = phase.min()
    S = numpy.zeros(N)
    averages = numpy.zeros(N, dtype=numpy.float32)

    for i in range(0,signal.shape[0]):
        averages[phase[i]-m]+=signal[i]
        S[phase[i]-m]+=1

    for i in range(0,S.shape[0]):
        if S[i]>0:
            averages[i]/=S[i]
        else:
            averages[i]=numpy.nan

    averages = pd.Series(averages).interpolate().values

    if timerange is None:
        sign_approx = averages[phase-m]
        sign_approx[numpy.where(phase < 0)] = -1
    else:
        sign_approx = averages[timerange-m]


    return sign_approx
# --------------------------------------------------------------------------------------------------------------------
def generate_signal_2d(H,W,A,freq):
    image = numpy.zeros((H,W))
    for r in range(H):
        image[r] = generate_signal(numpy.arange(0,W,1),A,freq)

    image-=image.min()
    image = image*255/image.max()
    return image
# --------------------------------------------------------------------------------------------------------------------
def generate_signal(time_vec, A, freq):
    noise = 0.05 * numpy.random.randn(time_vec.size)
    X = numpy.zeros(len(time_vec))
    for a,f in zip(A,freq):
        X+= a*numpy.sin(2*numpy.pi*f*time_vec)
    return X + noise
# ----------------------------------------------------------------------------------------------------------------------
def get_AF(X,time_step=1):
    if len(X.shape)==1:
        power = numpy.abs(fftpack.fft(X))
        sample_freq = fftpack.fftfreq(X.size, d=time_step)
    else:
        power,sample_freq = [],[]
        for c in range(X.shape[1]):
            x = X[:, c]
            power.append(numpy.abs(fftpack.fft(x)))
            sample_freq.append(fftpack.fftfreq(x.size, d=time_step))
        power = numpy.max(numpy.array(power),axis=0)
        sample_freq = numpy.max(numpy.array(sample_freq), axis=0)

    idx = numpy.where(sample_freq > 0)
    power = power[idx]
    sample_freq = sample_freq[idx]
    power = 255 * power / power.max()
    return sample_freq, power
# ----------------------------------------------------------------------------------------------------------------------

def plt_FFT_power(sample_freq, power,frequency_based=True,filename_out=None):

    plt.figure(figsize=(6, 5))
    if frequency_based:
        plt.plot(sample_freq, power)
        #plt.xlim(left=0,right=0.5)
        plt.xlabel('Frequency [Hz]')
    else:
        periods = 1/sample_freq
        plt.plot(periods, power)
        plt.xlim(left=1,right=30)
        plt.ylim(bottom=-0,top=260)
        plt.xlabel('Period')
        plt.grid()

    plt.ylabel('plower')
    if filename_out is not None:
        plt.savefig(filename_out)
    return
# ----------------------------------------------------------------------------------------------------------------------
def plt_signal(time_vec,X,filename_out=None):
    plt.figure(figsize=(6, 5))
    plt.plot(time_vec, X, label='Original signal')
    if filename_out is not None:
        plt.savefig(filename_out)
    return
# ----------------------------------------------------------------------------------------------------------------------
def get_powerful_freq(sample_freq, power):
    idx = numpy.argsort(-power)
    return numpy.abs(sample_freq[idx[0]])
# ----------------------------------------------------------------------------------------------------------------------