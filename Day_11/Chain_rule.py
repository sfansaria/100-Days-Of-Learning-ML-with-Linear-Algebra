'''
In Deep Learning, Error does not just flow through single numbers.
It flows backward through entire multidimensional tensors. 
The Chain Rule - It is the mathematical conveyor belt that passes error gradients
backward from tensor to tensor, layer to layer  for example you see this in the Transformer Architecture.

Let us learn chain  rule from the speech ml perspective as The "Telephone Game" of Error

Imagine you have a 3-layer speech network trying to recognize a voice command.
layer 1 (The Ear): Takes a raw audio tensors and filters out the background noise.
layer 2 (The Linguist): Takes the clean audio tensor and identifies phonetic sounds (like "sh" or "ah")
layer 3 (The Brain): Takes phonetics and predicts the final text word

If the model makes a mistake at the very end (layer 3), how does layer 1 know how to change its audio-filtering 
weights? layer 1 cannot see the final error directly.

This is where the Chain Rule comes in. It states that to find out how a change in layer 1 affects the final error, 
you simply "multiply the rates of change (derivatives) of the layers in between".

\(\frac{\partial \text{Total\ Error}}{\partial \text{Layer\ 1\ Weights}}=\frac{\partial \text{Total\ Error}}{\partial \text{Layer\ 3}}\times \frac{\partial \text{Layer\ 3}}{\partial \text{Layer\ 2}}\times \frac{\partial \text{Layer\ 2}}{\partial \text{Layer\ 1\ Weights}}\)

By mulltiplying these links together, the final error signal successfully travels all the way back
to the beginning of the model. This entire process is called Backpropagation.

In production frameworks like PyTorch, this chain of derivatives is trcaked automatically
using a hidden system called a Computational Graph. Everytime you perform a math on a tensor, 
PyTorch notes down the  derivative formula required to back up through that specific operation.

'''
import torch

#Forward
#layer1: simulated raw audio feature tensor (loudness level)
#lets see how tuning layer 1's weight impacts the final text error
audio_input = torch.tensor([1.5])
w_layer1 = torch.tensor([2.0], requires_grad=True)

#layer1 processes the audio
layer1_output = audio_input * w_layer1
layer1_output.retain_grad() #Ask PyTorch to save this intermediate step's gradient

#Layer 2 : Phonetic processing layer that squares signal
#Forward step 2 : Layer 2 processes layer 1's output
layer2_output = layer1_output ** 2 
layer2_output.retain_grad()

#final step to calculate the loss (error) value against a target value of 10.0
target = torch.tensor([10.0])
loss = (layer2_output - target)**2

#loss.backward() calculates the math from end to the start 
#calculus backward pass
#running backward() triggers the chain rule across the computational graph
loss.backward()

print("Chain Rule Components  - Backwards")

# Link 1: dLoss / dLayer2 (Derivative of power function)
#Link 1 calculates how the loss changes relative to Layer 2 (-2.0)
print(f"Link 1 (Loss to Layer 2 gradient) (thats is how loss changes relative to layer 2): {layer2_output.grad.item():.2f}")

# Link 2: dLoss / dLayer1 (Calculated as: Link 1 * Derivative of squaring step)
#The model then multiplies that by the derivative of the squaring function (2 × 3.0 = 6.0)
# which gives Link 2 (-2.0 × 6.0 = -12.0).
print(f"Link 2 (Loss down to Layer 1)(loss):      {layer1_output.grad.item():.2f}")

# Link 3: dLoss / dWeight1 (Calculated as: Link 2 * Raw audio input)
#Finally, it multiplies that result by the raw audio input (1.5) 
#to get the final gradient for our starting weight (-12.0 × 1.5 = -18.0).
print(f"Final Weight 1 Gradient: {w_layer1.grad.item():.2f}")


'''
Why chain rule is important is because without chain rule, deep learning 
neural networks would be completely blind, they could caluculate mistakes
at the output layer but they would have no mathematical way to pass that 
knowledge back to update the lower tensor features.
'''