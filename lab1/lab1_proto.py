# DT2119, Lab 1 Feature Extraction
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sis
import scipy.fftpack as sisfft
import lab1_tools as lab1tools
from more_itertools import windowed




example = np.load('lab1_example.npz', allow_pickle=True)['example'].item()
data = np.load('lab1_data.npz', allow_pickle=True)['data']


# Function given by the exercise ----------------------------------

def mspec(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps)
    return lab1tools.lifter(ceps, liftercoeff)

# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
    
    range(0, np.size(samples))    numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """

    #samples >= winshift*N+winlen

    #M = int((len(samples)-winshift)/winshift)
    """
    N = int((len(samples)-winshift)/winshift) #Lucas' fundamental theory of "att räkna ut windows"
    windows = np.zeros((N, winlen))
    windows_num = 0
    
    for i in range(0, len(samples), winshift):
        if i+winlen > len(samples):
            break

        window_iter = np.arange(i,i+winlen)
        windows[windows_num] = samples[window_iter]
        windows_num += 1
    
    """
    win = list(windowed(samples, winlen, fillvalue=0, step=winshift))
    return np.asarray(win)
    
    
def preemp(inputs, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:4
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """    


    A = [1]
    B = [1, -p]
    return sis.lfilter(B, A, inputs, axis=1)


def windowing(inputs):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    
    return inputs * sis.hamming(400, sym=False)


def powerSpectrum(inputs, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    return np.abs(sisfft.fft(inputs, nfft)) **2
    

def logMelSpectrum(inputs, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    fbank = lab1tools.trfbank(samplingrate, np.size(inputs, axis=1))
    return np.log(inputs @ fbank.T)

def cepstrum(inputs, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """

    #return sisfft.dct(inputs, n=nceps)
    return sisfft.dct(inputs)[:, :nceps]

def dtw(x, y, dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """
    return 1


#print(np.size(example['samples']))
#print(example['samples'])
#plt.plot(example['samples'])
#plt.show()
#plt.pcolormesh(example['frames'])
#plt.show()
#mspec(example['samples'])


windows = enframe(example['samples'], 400, 200)
#plt.plot(windows)
#plt.pcolormesh(windows)
#plt.show()


preemp_window = preemp(windows)
#plt.plot(preemp_window)
#plt.pcolormesh(preemp_window)
#plt.show()


windowing = windowing(preemp_window)
#plt.pcolormesh(windowing)
#plt.show()


pSpec = powerSpectrum(windowing,512)

#print(pSpec)
#plt.plot(pSpec)
#plt.pcolormesh(pSpec)
#plt.show()
#plt.pcolormesh(example['spec'])
#plt.show()

melSpec = logMelSpectrum(pSpec,20000)
#plt.pcolor(melSpec)
#plt.show()

cStrum = cepstrum(melSpec,13)
l = lab1tools.lifter(cStrum, 22)
plt.pcolor(example['lmfcc'])
plt.show()
plt.pcolor(l)
plt.show()