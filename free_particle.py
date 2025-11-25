
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation # Import the animation class

print("Running free particle animation...")

## 1. SETUP: Define Constants and Grids
hbar = 1.0
m = 1.0
N = 1024
L = 100.0
dx = L / N
x = np.linspace(-L/2, L/2, N)
k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
dt = 0.05
n_steps = 1000 # This will now be the number of frames

## 2. DEFINE: Initial State and Potential
x0 = - L / 4
k0 = 0
sigma = 2
psi0 = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
norm = np.sum(np.abs(psi0)**2) * dx
psi = psi0 / np.sqrt(norm) # 'psi' will be our global state variable
V = np.zeros(N)

## 3. PRE-CALCULATE: Evolution Operators
V_op = np.exp(-1j * V * dt / (2 * hbar))
T_op = np.exp(-1j * hbar * k**2 * dt / (2 * m))

## 4. SETUP THE ANIMATION

# --- Set up the plot ---
# 'fig' and 'ax' are now global so 'update' can access them
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the initial state (t=0)
# 'line,' gets the Line2D object, which we will update
line, = ax.plot(x, np.abs(psi)**2)

# Set plot limits and labels
ax.set_ylim(0, np.max(np.abs(psi)**2) * 1.2) # Set a fixed y-limit
ax.set_xlim(x[0], x[-1])
ax.set_xlabel("Position (x)")
ax.set_ylabel("Probability Density |ψ(x,t)|²")
ax.grid(True)
# Add a title that we can update
title = ax.set_title(f"Time = 0.00")

# --- Define the update function ---
# This function runs for every frame
def update(frame):
    global psi # Tell the function to use the global 'psi' variable
    
    # Run one step of the simulation
    # (a) First half potential step
    psi = psi * V_op
    
    # (b) Full kinetic step
    psi_k = np.fft.fft(psi)
    psi_k = psi_k * T_op
    psi = np.fft.ifft(psi_k)
    
    # (c) Second half potential step
    psi = psi * V_op
    
    # (d) Update the plot data
    prob_density = np.abs(psi)**2
    line.set_ydata(prob_density)
    
    # Update the title
    current_time = frame * dt
    title.set_text(f"Time = {current_time:.2f}")
    
    return line, title # Return the objects that were changed

## 5. RUN THE ANIMATION

# Create the animation
# 'frames' is the number of times to call 'update'
# 'blit=True' means it only redraws the parts that changed (faster)
ani = FuncAnimation(fig, update, frames=n_steps, blit=False)

# Show the plot
plt.show()