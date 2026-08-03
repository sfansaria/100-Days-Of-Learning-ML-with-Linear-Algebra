import numpy as np
import matplotlib.pyplot as plt

v_speech = np.array([3, 0]) #clean speech on x-axis 

v_noise = np.array([0, 4]) #background noise on y-axis
noisy_vector =v_speech + v_noise #recorded noisy speech

#create a plot figure to visalise the vectors

plt.figure(figsize=(6,6))
ax = plt.gca()

plt.quiver(0, 0, v_speech[0], v_speech[1], angles= 'xy', scale_units='xy', 
           color='green', label='Clean Speech Axis (Projection Target)')
plt.quiver(0, 0, v_noise[0], v_noise[1], angles='xy', scale_units='xy', 
           color='red', label='Distorted Input Audio Vector (Speech+Noise)')

plt.plot([v_noise[0], v_speech[0]], [v_noise[1], v_speech[1]], linestyle='--',
         color='blue', linewidth=2, label='Perpendicular Projection Line')

plt.plot(v_speech[0], v_speech[1], 'go', markersize=8)
plt.text(v_speech[0]-0.2, v_speech[1]-0.4, 'Cleaned Output\n(3,0)', color='green', fontweight='bold')
plt.text(v_noise[0]+0.1, v_noise[1]+0.1, 'Noisy Input\n(0, 4)', color='red', fontweight='bold')

plt.xlim(-1, 5)
plt.ylim(-1, 5)
plt.axhline(0, color='black', linewidth=0.8, linestyle=':')
plt.axvline(0, color='black', linewidth=0.8, linestyle=':')
plt.grid(True, which='both', linestyle=':', alpha=0.5)
plt.gca().set_aspect('equal', adjustable='box')

plt.title('Speech ML: Noise Filtering via Vector Projection', fomtsize=12, fontweight='bold', pad=15)
plt.xlabel('Speech Features Axis (Signal Importance)', fontsize=10)
plt.ylabel('Noise Features Axis (Perpendicular Interference)', fontsize=10)
plt.legend(loc='upper left', fontsize=9)

plt.show()