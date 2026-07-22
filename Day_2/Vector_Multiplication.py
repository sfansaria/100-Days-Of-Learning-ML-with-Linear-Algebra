'''Explaining the Vector multiplication  from Speech and audio perspective.
   Sound cannot be understood by the machine as a raw wave. Hence, it must be converted to vectors(sequences of numbers).
   Vector multiplication is the mathematical tool that transforms raw audio into text, 
   filters out the background noise, and enhances the quality of the sound and recognizes the voice.
   '''
#Scalar multiplication: Scaling the amplitude of the audio signal by multiplying it with a scalar value.

import numpy as np

#a short audio frame (e.g., a speech sample)
raw_speech_audio = np.array([0.15, -0.30, 0.45, -0.10, 0.05])

print("--- RAW AUDIO INPUT ---")
print(f"Speech Vector: {raw_speech_audio}\n")

voulme_gain = 2.5
amplified_audio = voulme_gain * raw_speech_audio
print (f"Amplified speech: {amplified_audio}")


#HADAMARD PRODUCT: Noise Gating / Masking

# Imagine the 3rd element in our vector is a sudden microphone hiss or noise spike.
# We apply a "mask vector" to silence that specific component.
noise_gate_mask = np.array([1.0, 1.0, 0.0, 1.0, 1.0])  # Zero out the 3rd index
cleaned_speech = amplified_audio * noise_gate_mask

print("--- HADAMARD PRODUCT (Noise Filtering) ---")
print(f"Mask Vector:     {noise_gate_mask}")
print(f"Cleaned Speech:  {cleaned_speech}")
print("Application: Removes static background noise or applies acoustic filters.\n")


#Vector Dot Product: 
#When the speech audio vector aligns perfectly with the target keyword vector,
# the dot product will be maximized, indicating a strong match. A high dot product means there is high acoustic similarity between the two vectors.
#on the other hand if the vectors are orthogonal (perpendicular), 
#the dot product will be zero, indicating the sound is unrelated to the target keyword and the word error rate will be high.
'''
Dot Product: The "Wake Word" Radar Lock

Imagine Batman's Batcomputer listening to the room, waiting for him to say his wake-word: "Computer".
The Cartoon Visual: Think of the target word "Computer" as a rigid mould or stencil carved out of wood. 
The incoming live audio wave is a flexible strip of metal sliding underneath it.
The Interaction:No Match: Batman says "Robin". The peaks of his voice hit the flat parts of the stencil. 
They don't fit. Positive numbers multiply by negative numbers, canceling each other out. 
The dot product stays near 0.

Perfect Match: Batman says "Computer". The sliding sound wave snaps perfectly into the stencil.
Every peak of his voice lines up with a peak in the stencil (Positive × Positive = Big Positive). 

Every valley lines up with a valley (Negative × Negative = Big Positive).
The Intuition: All the multiplications become positive and flood together into a massive sum. 
The dot product spikes like an alarm, telling the AI: "The pattern matches! Wake up!"
'''
#DOT PRODUCT: Wake-Word Detection ("Acoustic Alignment")
#The AI is listening for a specific phonetic target vector (e.g., the syllable "Ah")
wake_word_target = np.array([0.35, -0.75, 0.0, -0.25, 0.10])

# Compute similarity via the dot product
similarity_score = np.dot(cleaned_speech, wake_word_target)

print("--- DOT PRODUCT (Wake-Word Detection) ---")
print(f"Wake Word Reference: {wake_word_target}")
print(f"Acoustic Match Score: {similarity_score:.4f}")
print("Application: If this score clears a threshold, your smart speaker wakes up.")
