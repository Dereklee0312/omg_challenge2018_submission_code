"""Speech feature extraction utilities driven by shared STFT/sampling config."""

import numpy as np
import essentia.standard as ess
import essentia
import utilities_func as uf
from speech_config import speech_stft, speech_sampling

stft_cfg = speech_stft()
sampling_cfg = speech_sampling()

# Get STFT/sampling values from shared defaults-backed speech config.
WINDOW_SIZE = int(stft_cfg["window_size"])
FFT_SIZE = int(stft_cfg["fft_size"])
HOP_SIZE = int(stft_cfg["hop_size"])
WINDOW_TYPE = str(stft_cfg["window_type"])
SR = int(sampling_cfg["sr"])
fps = 25  #annotations per second
hop_annotation = SR /fps
frames_per_annotation = hop_annotation/float(HOP_SIZE)

def extract_features(x, M=WINDOW_SIZE, N=FFT_SIZE, H=HOP_SIZE, fs=SR, window_type=WINDOW_TYPE):
    """Return power-law-compressed magnitude spectra for an audio signal."""
    #init functions and vectors
    x = essentia.array(x)
    spectrum = ess.Spectrum(size=N)
    window = ess.Windowing(size=M, type=window_type)
    SP = []

    #compute STFT
    for frame in ess.FrameGenerator(x, frameSize=M, hopSize=H, startFromZero=True): #generate frames
        wX = window(frame)  #window frame
        mX = spectrum(wX)  #compute fft
        ###############################OPTIMIZATION[[[[[[[[[[[[[[]]]]]]]]]]]]]]
        #DEPRECATED
        #################################################
        SP.append(mX)

    SP = essentia.array(SP)
    SP = np.power(SP, 2./3.)  #power law compression

    return SP
