'''
Orthogonality and Projections: 

Orthogonality - Two vectors are orthogonal if they are perpendicular to each other and 
their dot product is zero. 
From speech machine learning perspective, Orthogonality means zero reduncy, that is the features 
are independent of each other. 

Let us consider an AI processing audio, where Axis X captures the speaker's pitch (high or low voice) and 
Axis Y captures the background room humm (constant low fan noise). As the fan noise doesn't change 
with the speaker's pitch, the two features are independent of each other and hence orthogonal.

Projections - It is taking a high-dimensional vector and dropping a perpendicular line to cast its "shadow"
onto a lower-dimensional subspace.

Let us consider a microphone recording a messy speech v_messy containing a speaker's voice and air plane rumble.
The AI defines a clean 1D line (subspace) that represents only human speech patterns. 
By projecting the messy vector onto the clean subspace, it extracts the clean voice "shadow" and drops the airplane noise
completely.

'''

import numpy as np

#define the clean speech direction vector (The subspace)
#mathematical axis for clean speech 
v_clean = np.array([3, 0]) 

#orthogonal noise vector (The noise subspace)
#this vector sits at a perfect 90-degree angle to the clean speech vector
v_noise = np.array([0, 2])

print("Dot Product (Check the Orthogonality):", np.dot(v_clean, v_noise)) #should be 0, confirming orthogonality

#Messy Distorted Speech Vector (real-world audio noise)
v_messy = v_clean + v_noise

#Vector projection of the messy speech onto the clean speech subspace
#proj_u(v) = ((v*u)/(u*u))*u

dot_product_v_u = np.dot(v_messy, v_clean)
dot_product_u_u = np.dot(v_clean, v_clean)

projection_scalar = dot_product_v_u / dot_product_u_u

v_cleaned = projection_scalar * v_clean

print("\n Speech Processing Results\n:")
print("Messy Input Audio Vector:", v_messy)
print("Cleaned Audio Output:", v_cleaned)

