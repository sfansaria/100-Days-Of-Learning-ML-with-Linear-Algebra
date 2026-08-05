'''
Scalar (Rank 0) : A single number for example: 0.8
Vector (Rank 1) : A 1D list of numbers for example: a single 25 ms of audio frame
Matrix (Rank 2) : A 2D grid of numbers for example: a spectogram of an entire audio file
Tensor (Rank 3 or higher) : A 3D or higher block of numbers for example: a batch of 32 differenr audio 
spectograms packaged together to train a model simultaneously

Why tensor matters immensely in Speech ML: Speech data is rarely processed one single file at a time.
It is processed in 3D or 4D Tensors. 
Let us consider loading data into a PyTorch model.
- Dimension 1 - Batch size: The number of audio clips processing at the exact same millisecond. (16 files)
- Dimension 2 - Time/Frames: The length of the audio file chopped into slices. (500 frames)
- Dimension 3 - Features/Frequencies: The acoustic components of each frame. (80 Mel-frequency channels)

This creates a 3D Tensor with the shape [16, 500, 80]

'''

import torch

batch_size = 16 #16 audio files being processed at once
time_frames = 500 #Each file 500 time slices long
audio_features = 80 #Each file has 80 frequency bins (Mel-spectogram)

#Creating a random tensor to simulate this batch of audio data
speech_tensor = torch.randn(batch_size, time_frames, audio_features)

print("PyTorch Tensor Diagnostics")
print("Tensor Shape: ", speech_tensor.shape)
print("Number of Dimensions:", speech_tensor.ndim)

#Accessing specific slices of data
#Extract the 1st audio file, the 50th time frame, and look at all its frequency features
single_frame_vector = speech_tensor[0, 49, :]
print("Extracted Slice Shape:", single_frame_vector.shape) #Becomes a 1D Vector again
