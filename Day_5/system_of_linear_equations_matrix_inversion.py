'''
Let us consider from the speech perspective, during audio processing, 
using the math of system of linear equations for "working  backwards" 
solving the problem of deconvolution (reversing the channel distortion).
Imagining a voice is recorded inside a large, empty hall.
let the clean, original voice vector be 'x'.
let the echoey acoustices of the physical room act as a transformation matrix be 'A'.
let the noisy, echoed audio be 'b'  recorded on the microphone.
we get to know speech recognition model works accurately if it removes the echo.
lets set up a system of linear equations: Ax = b

In order to get a clean voice x, the computer computes the inverse matrix A-1 to 
undo the room's transformation: x = (A^-1)*b


The Machine Learning Reality: Overdetermined Systems: 
In an ideal math textbook, you have the exact same number of equations as unknowns, 
and you find a perfect solution. In machine learning, you face overdetermined systems.
You have millions of audio samples (equations) but only a few thousand model parameters (unknowns). 
A perfect solution does not exist.Because of this, this topic introduces the 
Ordinary Least Squares (OLS) method and the Pseudo-inverse. Instead of finding a perfect solution, 
the math calculates a line of best fit that minimises the total error across the entire dataset. 
This is the foundation of training a Linear Regression model.
'''

import numpy as np

A = np.array([[1, 3], [4, 5]]) #The room's acoustic distortion properties

b = np.array([11, 13]) #noisy, distorted audio recorded by the microphone

#Reconstruct the clean, original speeech signal x = (A^-1)*b

x = np.linalg.solve(A, b)
print(f"Reconstructed voice vector x: {x}")
