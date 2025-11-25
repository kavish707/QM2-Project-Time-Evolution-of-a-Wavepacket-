import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# Use this if you are in Jupyter to embed the video
from IPython.display import HTML

print("Running potential barrier simulation...")

## 1. SETUP: Define Constants and Grids
hbar = 1.0
m = 1.0
N = 4096
L = 100.0
dx = L / N
x = np.linspace(-L/2, L/2, N)
k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
dt = 0.01
n_steps = 1000

## 2. DEFINE: Initial State and Potential

# Initial wave packet parameters
x0 = -L / 4             # Initial position
k0 = 10                # Initial momentum
sigma = 1            # Width of the wave packet

# Define the initial wave packet
psi0 = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
norm = np.sum(np.abs(psi0)**2) * dx
psi = psi0 / np.sqrt(norm)

# --- NEW: Define the Potential V(x) ---
V = np.zeros(N)
barrier_height = 2000.0
barrier_width = 1.0
barrier_center = 0.0

# Create the barrier
V[ (x > barrier_center - barrier_width/2) & (x < barrier_center + barrier_width/2) ] = barrier_height


## 3. PRE-CALCULATE: Evolution Operators

# --- NEW: Recalculate V_op with the new V ---
V_op = np.exp(-1j * V * dt / (2 * hbar))

# Kinetic operator is unchanged
T_op = np.exp(-1j * hbar * k**2 * dt / (2 * m))


## 4. SETUP THE ANIMATION PLOT

fig, ax = plt.subplots(figsize=(10, 6))

# Plot the potential barrier (scaled to fit)
# This gives a visual reference
ax.plot(x, V / barrier_height * 0.5, 'r--', label="Potential Barrier (scaled)")

# Plot the initial state
line, = ax.plot(x, np.abs(psi)**2, label="|ψ(x,t)|²")

# Set plot limits and labels
ax.set_ylim(0, np.max(np.abs(psi)**2) * 1.2)
ax.set_xlim(x[0], x[-1])
ax.set_xlabel("Position (x)")
ax.set_ylabel("Probability Density")
ax.grid(True)
title = ax.set_title(f"Time = 0.00")
ax.legend()

# --- Define the update function ---
# This is identical to the free particle case
def update(frame):
    global psi 
    
    psi = psi * V_op
    psi_k = np.fft.fft(psi)
    psi_k = psi_k * T_op
    psi = np.fft.ifft(psi_k)
    psi = psi * V_op
    
    prob_density = np.abs(psi)**2
    line.set_ydata(prob_density)
    
    current_time = frame * dt
    title.set_text(f"Time = {current_time:.2f}")
    
    return line, title 

## 5. RUN THE ANIMATION

# Create the animation object
ani = FuncAnimation(fig, update, frames=n_steps, blit=False)

# To display in Jupyter Notebook:
# print("Rendering animation...")
# plt.close(fig) # Close the static plot
# video_html = ani.to_html5_video()
# HTML(video_html)

# To display by running as a .py file:
plt.show()