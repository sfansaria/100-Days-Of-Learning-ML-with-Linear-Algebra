# A Vector - It is a data point for example in terms of speech, it is an audio features of one spoken word
# A Matrix - It is an entire dataset for example in terms of speech , it is the audio features of hundreds of spoken words
# stacked on top of each other.

# Now lets understand linear transformation in from speech machine learning perspective,
# a linear transformation is a mathematical operation that reshapes, filters or compresses 
# audio data to make it easier for an AI to understand. When the udio is recorded, it is highly complex, noisy and raw. A linear transformation takes these raw speech features and transforms them into a cleaner, more organized structure without bending or destroying the underlying relationships in the data.

# From the speech context: Before transforming speech, the AI chops your continuous voice recording into short, overlapping slices called frames (usually 25 milliseconds long).

# Each frame is converted into a vector of numbers representing different acoustic frequencies.
# When you stack these frame vectors side-by-side, they form a matrix called spectogram (a visual map of the voice).

# Linear Transformations are used at almost every stage of speech processing.

# 1. Feature Extraction (The Mel-Filterbank):

# Human ears do not hear pitch shifts linearly; but are much better at detceting small changes in low frequencies
# than in high frequencies.
# A raw audio vector goes through a linear transformation matrix called a Mel-filterbank
# This mel-filterbank compresses thousands of raw frequencies down to a small, biologocally accuracte vector (typically 40 or 80 dimensions). The matrix stretches the critical low pitched zones and squeezes the high-pitched zones.

# 2. Noise Reduction and Channel Normalization:

# If a voice is recorder while driving a car, then the vector contains both the voice and the low-frequency rumble of the engine.
# A tranformation matrix can project the speech vector onto a new set of mathematical axes. This aligns the engine noise along one axis and the clean voice along another, allowing the AI to safely isolate and
# ignore the noise axis.

# 3. Neural Network Hidden Layers (Acoustic Modelling)

# When speech vector pass through the layers of an automatic speech recognition (ASR) model, they undergo repeated linear transformations.
# Each layer multiplies the incoming speech vector by a trained weight matrix. This warps the data space. 
# It pulls different voices speaking the same word (such as a word "he" said by both Adult and Child) closer together in space,
# while pushing words that are different further apart such as "He" and "She" . 

import numpy as np

speech_v = np.array([3.5, 4.2, 8.0]) #contains [low, mid, high] frequencies
#keeping the 100 5 of low and mid frequencies but compressing the high


#Transformation matrix
filter_matrix = np.array([[1.0, 0.0, 0.0], 
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.1]])

#Transforming the speech by applying linear transformation that is simply a matrix vector multiplication

transformed_speech = np.dot(filter_matrix, speech_v)

print("Original Raw Speech Vector: ", speech_v)
print("Cleaned Speech vector: ", transformed_speech)