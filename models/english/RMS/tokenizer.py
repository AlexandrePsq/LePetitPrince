import wave
import numpy as np
import os



def tokenize(path, slice_period=100e-3):
    # slice_period is in s
    wave_file = wave.open(path, mode='r')
    rate = wave_file.getframerate()
    n_frames = wave_file.getnframes()   # Number of frames.
    slice_length = slice_period * rate


    # Read audio data.
    data = np.fromstring(wave_file.readframes(n_frames), dtype=np.int16)
    data_list = [data[index*slice_length: index*slice_length + slice_length] for index in range(n_frames//slice_length)]
    return data_list
