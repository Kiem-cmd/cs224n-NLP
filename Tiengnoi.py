import numpy as np
import scipy.io.wavfile as wavfile


sample_rate, data = wavfile.read("a96.wav")

data = data[::2]  
data = data[data.size//4:]  

autocorr = np.correlate(data, data, mode="full")


peak = np.argmax(autocorr)


F0 = sample_rate / peak

print("F0 (T0) bằng phương pháp PP tự tương quan:", F0)
