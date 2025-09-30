import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def uniform_polar_walk(n, d=1, theta_mean=0, theta_range=None, tqdm_bool=False):
    """
    Model 1: Continuous uniform distribution of angles
    """
    x = np.zeros(n)
    y = np.zeros(n)
    
    if theta_range is None:
        # Full uniform distribution [0, 2π] as per instructions
        theta_min, theta_max = 0, 2*np.pi
    else:
        # Biased uniform (original function behavior)
        theta_min, theta_max = theta_mean - 2*theta_range, theta_mean + 2*theta_range
    
    for k in tqdm(range(1, n), desc=f'Walking {n} Steps', disable=not tqdm_bool):
        theta = np.random.uniform(theta_min, theta_max)
        x[k] = x[k-1] + d * np.cos(theta)
        y[k] = y[k-1] + d * np.sin(theta)
    
    return x, y

def four_direction_walk(n, d=1, tqdm_bool=False):
    """
    Model 2: Four possible directions (0, π/2, π, -π/2) with equal probability
    """
    x = np.zeros(n)
    y = np.zeros(n)
    directions = [0, np.pi/2, np.pi, -np.pi/2]
    
    for k in tqdm(range(1, n), desc=f'Walking {n} Steps (4 directions)', disable=not tqdm_bool):
        theta = np.random.choice(directions)
        x[k] = x[k-1] + d * np.cos(theta)
        y[k] = y[k-1] + d * np.sin(theta)
    
    return x, y

def plot_trajectories(M=10, N=100):
    """
    Plot M trajectories for each model (Task 1)
    """
    plt.figure(figsize=(15, 6))
    
    # Model 1: Continuous uniform
    plt.subplot(1, 2, 1)
    for i in range(M):
        x, y = uniform_polar_walk(N)
        plt.plot(x, y, alpha=0.7, linewidth=1)
    plt.plot(0, 0, 'ro', markersize=8, label='Start')
    plt.title(f'Model 1: {M} Random Walks (N={N})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Model 2: Four directions
    plt.subplot(1, 2, 2)
    for i in range(M):
        x, y = four_direction_walk(N)
        plt.plot(x, y, alpha=0.7, linewidth=1)
    plt.plot(0, 0, 'ro', markersize=8, label='Start')
    plt.title(f'Model 2: {M} Random Walks (N={N})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def verify_root_mean_square(N_values, d=1, num_samples=1000):
    """
    Verify that √<x_N² + y_N²> = √N * d (Task 2)
    """
    rms_model1 = []
    rms_model2 = []
    
    for N in tqdm(N_values, desc='Calculating RMS'):
        # Model 1
        distances1 = []
        for _ in range(num_samples):
            x, y = uniform_polar_walk(N, d)
            distance = np.sqrt(x[-1]**2 + y[-1]**2)
            distances1.append(distance)
        rms_model1.append(np.sqrt(np.mean(np.array(distances1)**2)))
        
        # Model 2
        distances2 = []
        for _ in range(num_samples):
            x, y = four_direction_walk(N, d)
            distance = np.sqrt(x[-1]**2 + y[-1]**2)
            distances2.append(distance)
        rms_model2.append(np.sqrt(np.mean(np.array(distances2)**2)))
    
    # Theoretical prediction
    theoretical = np.sqrt(N_values) * d
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(N_values, rms_model1, 'bo-', label='Model 1 (Continuous)', alpha=0.7)
    plt.plot(N_values, rms_model2, 'ro-', label='Model 2 (4 directions)', alpha=0.7)
    plt.plot(N_values, theoretical, 'k--', label=r'Theoretical: $\sqrt{N} \cdot d$', linewidth=2)
    
    plt.xlabel('Number of Steps (N)')
    plt.ylabel(r'$\sqrt{\langle x_N^2 + y_N^2 \rangle}$')
    plt.title('Verification of Random Walk Scaling Law')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return rms_model1, rms_model2, theoretical

# Execute the tasks
if __name__ == "__main__":
    # Task 1: Plot trajectories
    print("Task 1: Plotting trajectories for M=10, N=100")
    plot_trajectories(M=10, N=100)
    
    # Task 2: Verify scaling law
    print("\nTask 2: Verifying scaling law")
    N_values = np.arange(10, 201, 10)  # N from 10 to 200 in steps of 10
    rms1, rms2, theory = verify_root_mean_square(N_values)
