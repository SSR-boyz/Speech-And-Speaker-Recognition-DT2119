import numpy as np
import os
#from pysndfile import sndio
import soundfile as sf

def path2info(path):
    """
    path2info: parses paths in the TIDIGIT format and extracts information
               about the speaker and the utterance

    Example:
    path2info('tidigits/disc_4.1.1/tidigits/train/man/ae/z9z6531a.wav')
    """
    rest, filename = os.path.split(path)
    rest, speakerID = os.path.split(rest)
    rest, gender = os.path.split(rest)
    digits = filename[:-5]
    repetition = filename[-5]
    return gender, speakerID, digits, repetition

def loadAudio(filename):
    """
    loadAudio: loads audio data from file using pysndfile

    Note that, by default pysndfile converts the samples into floating point
    numbers and rescales them in the range [-1, 1]. This is avoided by specifying
    the option dtype=np.int16 which keeps both the original data type and range
    of values.
    """
    #sndobj = sndio.read(filename, dtype=np.int16)
    #samplingrate = sndobj[1]
    #samples = np.array(sndobj[0])
    return sf.read(filename, dtype='int16')

def frames2trans(sequence, outfilename=None, timestep=0.01):
    """
    Outputs a standard transcription given a frame-by-frame
    list of strings.

    Example (using functions from Lab 1 and Lab 2):
    phones = ['sil', 'sil', 'sil', 'ow', 'ow', 'ow', 'ow', 'ow', 'sil', 'sil']
    trans = frames2trans(phones, 'oa.lab')

    Then you can use, for example wavesurfer to open the wav file and the transcription
    """
    sym = sequence[0]
    start = 0
    end = 0
    trans = ''
    for t in range(len(sequence)):
        if sequence[t] != sym:
            trans = trans + str(start) + ' ' + str(end) + ' ' + sym + '\n'
            sym = sequence[t]
            start = end
        end = end + timestep
    trans = trans + str(start) + ' ' + str(end) + ' ' + sym + '\n'
    if outfilename != None:
        with open(outfilename, 'w') as f:
            f.write(trans)
    return trans

        
