#import wave
import scipy.io.wavfile as wave
import numpy as np



def tokenize(path, slice_period):
    # slice_period is in s
    # wave_file = wave.open(path, mode='r')
    # rate = wave_file.getframerate()
    # n_frames = wave_file.getnframes()   # Number of frames.
    # slice_length = int(slice_period * rate)
    [rate, data] = wave.read(path)
    slice_length = int(slice_period * rate)
    data_list = [np.array(data[index*slice_length: index*slice_length + slice_length], dtype=np.float64) for index in range(len(data)//slice_length)]
    

    # Read audio data.
    # data = np.frombuffer(wave_file.readframes(n_frames), dtype=np.int16)
    # data_list = [data[index*slice_length: index*slice_length + slice_length] for index in range(n_frames//slice_length)]
    return data_list, rate, len(data), slice_length
