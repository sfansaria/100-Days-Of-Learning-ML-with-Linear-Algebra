'''
Intuition OF **Vectors** connection to real world and how they are used in ML.
Lets consider in speech and audio processing, a  vector is represented as Voice Profile,(embeedding). 
Where each element of the vector represents a feature of the voice, such as pitch, ton, accent or background
noise. '''

'''Imagine you want a voice to sound like Dr. Sheldon Cooper from the Big Bang Theory. 
(Voice Covenrsion). AI model breaks down the voice into a vector representation,
capturing the unique charateristics of the voice. Then the model manipulates the vector 
to match the desired voice profile, resulting in a transformed voice that resembles Dr. Sheldon Cooper. 
So, here simply one vector is transformed into another vector.'''

import numpy as np

#normal voice profile vector
vec_normal = np.array([0, 0])  #Average pitch and Average human emotion
vec_sheldon = np.array([-4, 6]) #Lower pitch and Higher emotion

vec_trransformed = vec_normal + vec_sheldon

print("Normal Voice Profile Vector:", vec_normal)
print("Sheldon Voice Profile Vector:", vec_sheldon)
print("Transformed Voice Profile Vector:", vec_trransformed)