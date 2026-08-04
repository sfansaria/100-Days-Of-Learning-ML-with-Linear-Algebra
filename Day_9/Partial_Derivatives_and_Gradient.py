'''
In machine learning, a model relies on multiple features at the same time such as volume (loudness),
pitch(frequency), duration (length of a sound) to idetify a spoken word.
All the fetures interact at once, you cannot use a basic, single-variable calculus.
In order to isolate and study them, you must use Multivariate Calculus.

Partial Derivatives (Isolating the sound feature) - It measures how the model's total prediction
error changes if you tweak only one setting while keeping all the other settings completely locked in place. 

let us consider that the model is trying to recognize a word "Stop". The The speaker voice is too slow and 
the model fails to recognize that. The model calculates the partial derivative wrt Volume weight. 
"If the pitch and duration settings remains the same but increase the volume amplification, how much will the 
recognition error drop?"

Gradient - It is simply a vector that packages every single partial derivative together into one single block.
Geometrically, the gradient points in the direction of the steepest uphill slope on the error landscape.
During training the model, the optimization algorithm computes this gradient vector and takes a step in the 
exact opposite direction (downhill). This reduces the error across all the audio features simultaneously.

Uisng PyTorch in production speech ML (like Whisper or Siri), as the derivates are calculated by hand. 
Deep learning framework like PyTorch, which has a built-in automatic differentiation engine called Autograd.

lets simulate a simple speech node evaluating an audio snippet based on two features: Pitch and Volume . 
It calculates the loss (error) and uses PyTorch to automatically compute the partial derivatives (the gradient)
to see how to fix the model.

'''

import torch 

#Raw fetures of the spoken word snippet
#lets say: Pitch = 150.Hz, Volume = 0.8 (normalized)
pitch_input = torch.tensor(150.0)
volume_input = torch.tensor(0.8)

#Weights: The internal AI settings, in order to optimize, set requires_grad=True, because
#PyTorch calculates the calculus on them

w_pitch = torch.tensor(0.04, requires_grad=True)
w_volume = torch.tensor(2.0, requires_grad=True)

#Expected Target: the correct linguistic value the model should output
target_output = torch.tensor(10.0)

prediction = (w_pitch * pitch_input) + (w_volume * volume_input)
#Loss Function: Mean Squared
loss = (prediction-target_output)**2 

#Backward pass (The Calculus)- this single command tells PyTorch 
#to calculate all the partial derivatives

loss.backward()

#Extract the Gradient Vector
print("Speech ML Gradient Analytics")
print(f"Model Prediction: {prediction.item():.2f} (Target was {target_output.item()})")
print(f"Total Current Error: {loss.item():.2f}")

print("The Calculated Gradients")
print(f"Partial Derivative wrt Pitch Weight (dLoss/dw_pitch) : {w_pitch.grad.item():.2f}")
print(f"Partial Derivative wrt Volume Weight (dLoss/dw_volume) : {w_volume.grad.item():.2f}")