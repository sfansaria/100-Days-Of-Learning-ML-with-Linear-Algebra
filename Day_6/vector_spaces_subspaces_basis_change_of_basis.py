'''
-----------------------------------------------------------------------------------------------------
In order to understand the concept of vector spaces, let us first define,
vector spaces are mathematimatical structures formed by a collection
of vectors, which can be added together, or multiplied by a scalar value to scale up or down.
A subspace is a smaller space within a vector space. for example, a flat 2D sheet of paper
floating in a 3D room is a 2D subspace of a 3D space.
------------------------------------------------------------------------------------------------------
From the speech perspective, if you record a 10-second clip  at a standard sampling rate, 
your raw audio vector lives in an incredible massive 160,000 dimensional vector space.
Human speech sounds don't use all those dmensions randomly. Clean human voices only occupy 
a smaller, tightly constrained subspace inside that massive noise-filled space.
-----------------------------------------------------------------------------------------------------

Basis - It is a minimal set of independent vectors that can be combined to represent every vector
in the space and acts as the "backbone" or "coordinate system" for the vector space. 
Every single point in that space can be represented as a linear combination of the basis vectors.
Lets us think of a standard Cartesian grid:
i-hat = [1 0], and j-hat = [0 1] are the standard basis vectors for a 2D space.
Any 2D coordinate can be written as a linear combination of these two basis vectors.

For example [3 4] = 3*[1 0] + 4*[0 1] = 3*i-hat + 4*j-hat

-----------------------------------------------------------------------------------------------------

Change of Basis from the perspective of speech ml, means changing the perspective or shifting
to a brand new coordinate system where the data suddenly makes more sense.

Here comes the application of fourier transform, which is a mathematical technique that 
transforms a signal from its original domain Amplitude vs Time (the time-domain basis). 
This tells when a sound is produced it is incredebly difficult for an AI to understand
what word was spoken just by looking at a jagged waveform. To make sense of the sound, 
a linear change of basis is applied to the signal such as Fourier Transform, during this
the coordinate system is rotated into a new basis, the Amplitude vs Frequency (the frequency-domain
basis). This new basis is more suitable for speech recognition tasks, as it allows the AI to analyse.

The speech vector hasn't changed, but your coordinate system has. 
In this new basis, chords break down into individual notes, vowels separate into distinct frequencies,
and background noise becomes instantly visible and easy to crop out.

A change of basis means taking a vector and looking at it through a new set of coordinate axes.
Change of baisi using the Discrete Cosine Transform (DCT). This is the foundational step used to 
compute Mel-Frequency Cpestral Coefficients in speech recognition. It roates a frequency vector 
into a brand-new space where audio energy is concentrated in the first few coefficients, making 
it easier for the machine learning model to analyse and classify the sound. 
'''

import numpy as np
import scipy.fftpack as fftpack

#A speech frequency vector (energy across 4 audio bins)
frequency_vector = np.array([12.5, 3.2, 8.7, 1.1])

#perform a change of basis using a Discrete Cosine Trnasform (DCT)
#Under the hood, this multiplies the frequency vector by a new matrix of basis vectors
dct_basis_vector = fftpack.dct(frequency_vector, norm = 'ortho')
print("Original Basis (Frequency Energy) :", frequency_vector)
print("New Basis (DCT Cepstral Space) :", np.round(dct_basis_vector, 2))

