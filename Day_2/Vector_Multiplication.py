'''Explaining the Vector multiplication  from Speech and audio perspective.
   Sound cannot be understood by the machine as a raw wave. Hence, it must be converted to vectors(sequences of numbers).
   Vector multiplication is the mathematical tool that transforms raw audio into text, 
   filters out the background noise, and enhances the quality of the sound and recognizes the voice.
   '''
#Vector Dot Product: 
#When the speech audio vector aligns perfectly with the target keyword vector,
# the dot product will be maximized, indicating a strong match. A high dot product means there is high acoustic similarity between the two vectors.
#on the other hand if the vectors are orthogonal (perpendicular), 
#the dot product will be zero, indicating the sound is unrelated to the target keyword and the word error rate will be high.

import numpy as np

#a short audio frame (e.g., a speech sample)
raw_speech_audio = np.array([0.15, -0.30, 0.45, -0.10, 0.05])

print("--- RAW AUDIO INPUT ---")
print(f"Speech Vector: {raw_speech_audio}\n")

voulme_gain = 