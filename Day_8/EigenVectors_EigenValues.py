'''
In Speech ML, audio features containes hundreds of dimensions (frequencies), many of these
features are highly redundant or purely background noise. Principal Component Analysis (PCA) allows to 
extracts the core mathematical directions (EigenVectors) conatining the most vocal information
(EigenValues), effectively compressing the audio. 

EigenVectors (Distinct Vocal Shapes) - Considering a massive cloud of speech data points, 
the eigenvectors are the directional axes that point directly through the longest, thickest parts 
of the data cloud. In speech, one eigenvector might align perfectly with the variance in vowel resonance, 
while another tracks voice pitch.

EigenValues (Important Score) - Every Eignevector has a corresponding eigen values. 
It acts a score showing exactly how much data variance (information) is packed along that axis.
A high value - represents the feature is significant for the AI model, 
A low value near zero - represents that the axis contains useless ambient static.

Let us simulate a dataset of 100 Speech samples across 3 different frequency bins. 
At first calculate the covariance matrix, then extract the eigenvectors and use them to 
compress the 3D audio data into highly informative 2D space.

'''

import numpy as np
'''
let us generate a mock speech dataset (100 audio smaples,3 frequency diemnsions)
then add a pattern where 1 and 2 dimensions are highly correlated (the real voices) 
and dimension 3 is random low-energy background static noise.
'''

np.random.seed(42)
voice_signal = np.random.normal(0, 2, (100, 1))
noise_signal = np.random.normal(0, 0.5, (100, 1))

#combine the voice_signal and noise_signal to get a 3D matrix data [low_freq, mid_freq, high_freq_static]
speech_data = np.hstack([voice_signal, voice_signal*0.8 + np.random.normal(0, 0.5, (100, 1)), noise_signal]) 

#center the data around the origin (mean centering is required for PCA)
speech_data_centered = speech_data - np.mean(speech_data, axis=0)

#Calculate the covariance matrix (How frequencies change together)

covariance_matrix = np.cov(speech_data_centered, rowvar=False)

#Calculate the Eigenvectors and eigenvalues
e_val, e_vec = np.linalg.eig(covariance_matrix)

#Sort them from larget eigenvalues (most information) to smallest
sorted_indices = np.argsort(e_val)[::-1]
e_val = e_val[sorted_indices]
e_vec = e_vec[:, sorted_indices]

#compress to drop the noisy 3rd axis
top_2_e_vec = e_vec[:, :2]

#Project the original 3D speech data down into a clean 2D space
compressed_speech_2d = np.dot(speech_data_centered, top_2_e_vec)

print("PCA Speech Compression Summary")
print("Original Data Space: ", speech_data.shape)
print("Compressed Data Shape: ", compressed_speech_2d.shape)
print("\nInformation retained per axis (eigenvalues):")

for i, val in enumerate(e_val):
    print(f"Principal Component: {i+1}, Eigen value: {val:.4f}")
