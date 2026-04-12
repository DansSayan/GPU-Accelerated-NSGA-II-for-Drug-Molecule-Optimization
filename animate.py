import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load saved history
with open("generation_history.pkl", "rb") as f:
    history = pickle.load(f)

fig, ax = plt.subplots()

def update(frame):
    ax.clear()
    data = history[frame]
    
    # MolWt vs LogP (choose any 2 features)
    ax.scatter(data[:, 0], data[:, 1])
    
    ax.set_xlabel("MolWt")
    ax.set_ylabel("LogP")
    ax.set_title(f"Generation {frame+1}")

ani = FuncAnimation(fig, update, frames=len(history), interval=300)

# Save animation
ani.save("nsga_evolution.gif", writer="pillow")

print("Saved: nsga_evolution.gif")