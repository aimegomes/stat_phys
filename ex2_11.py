import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def box_random_walk(k, d=1, max_steps=100000, num_walks=1000):
    """
    Random walk in a square box with an exit hole
    k: parameter determining box size a = (2k+1)*d
    d: step size
    max_steps: maximum steps before giving up
    num_walks: number of random walks to average over
    """
    a = (2*k + 1) * d  # box side length
    half_box = a / 2
    
    # Hole position: (x_h = (k + 1/2)*d, y_h = 0)
    x_hole = (k + 0.5) * d
    y_hole = 0
    hole_size = d
    
    steps_to_exit = []
    directions = [0, np.pi/2, np.pi, -np.pi/2]  # Model 2: four directions
    
    for walk in tqdm(range(num_walks), desc=f'k={k}'):
        # Initial position uniformly distributed inside the box
        x = np.random.uniform(-half_box, half_box)
        y = np.random.uniform(-half_box, half_box)
        
        steps = 0
        inside_box = True
        
        while inside_box and steps < max_steps:
            # Generate potential step
            theta = np.random.choice(directions)
            x_new = x + d * np.cos(theta)
            y_new = y + d * np.sin(theta)
            
            # Check if new position is inside box boundaries
            if (-half_box <= x_new <= half_box) and (-half_box <= y_new <= half_box):
                # Valid step inside box
                x, y = x_new, y_new
                steps += 1
                
                # Check if exited through the hole
                if (abs(y - y_hole) < d/2 and 
                    abs(x - x_hole) < hole_size/2 and
                    x_new >= half_box):  # Moving out through the right side
                    inside_box = False
                    
            # If step would hit boundary, reject and try again (no step count increment)
            # We just generate a new direction in the next iteration
        
        if steps < max_steps:
            steps_to_exit.append(steps)
        else:
            # If max steps reached, discard this walk for average
            pass
    
    return np.mean(steps_to_exit) if steps_to_exit else max_steps

def analyze_escape_time(k_values, num_walks=500):
    """
    Analyze average escape time as function of k
    """
    avg_steps = []
    
    for k in k_values:
        avg_step = box_random_walk(k, num_walks=num_walks)
        avg_steps.append(avg_step)
        print(f"k={k}: Average steps to escape = {avg_step:.2f}")
    
    return avg_steps

def plot_single_trajectory(k, d=1):
    """
    Plot a single trajectory to visualize the behavior
    """
    a = (2*k + 1) * d
    half_box = a / 2
    x_hole = (k + 0.5) * d
    y_hole = 0
    
    # Create figure
    plt.figure(figsize=(10, 10))
    
    # Draw box boundaries
    plt.plot([-half_box, half_box], [half_box, half_box], 'k-', linewidth=2)
    plt.plot([-half_box, half_box], [-half_box, -half_box], 'k-', linewidth=2)
    plt.plot([half_box, half_box], [-half_box, half_box], 'k-', linewidth=2)
    plt.plot([-half_box, -half_box], [-half_box, half_box], 'k-', linewidth=2)
    
    # Draw hole (gap in the boundary)
    hole_start = x_hole - d/2
    hole_end = x_hole + d/2
    plt.plot([hole_start, hole_end], [half_box, half_box], 'w', linewidth=3)
    plt.plot([hole_start, hole_end], [-half_box, -half_box], 'w', linewidth=3)
    
    # Generate and plot a single trajectory
    directions = [0, np.pi/2, np.pi, -np.pi/2]
    x = np.random.uniform(-half_box, half_box)
    y = np.random.uniform(-half_box, half_box)
    
    trajectory_x = [x]
    trajectory_y = [y]
    max_steps = 10000
    
    for step in range(max_steps):
        theta = np.random.choice(directions)
        x_new = x + d * np.cos(theta)
        y_new = y + d * np.sin(theta)
        
        if (-half_box <= x_new <= half_box) and (-half_box <= y_new <= half_box):
            x, y = x_new, y_new
            trajectory_x.append(x)
            trajectory_y.append(y)
            
            # Check exit condition
            if (abs(y - y_hole) < d/2 and 
                abs(x - x_hole) < d/2 and
                x_new >= half_box):
                break
        # Else: hit boundary, try again
    
    # Plot trajectory
    plt.plot(trajectory_x, trajectory_y, 'b-', alpha=0.7, linewidth=1)
    plt.plot(trajectory_x[0], trajectory_y[0], 'go', markersize=8, label='Start')
    plt.plot(trajectory_x[-1], trajectory_y[-1], 'ro', markersize=8, label='Exit')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Random Walk in Box (k={k}, a={(2*k+1)*d})')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    return len(trajectory_x) - 1  # Return number of steps

def theoretical_analysis(k_values):
    """
    Provide theoretical expectations for comparison
    For a 2D random walk in a box of size L, the escape time scales with L²
    Since L = (2k+1)*d, we expect escape time ~ O(k²)
    """
    # Simple quadratic model: τ ~ C * k²
    theoretical = [10 * k**2 for k in k_values]  # Arbitrary scaling factor
    return theoretical

# Main execution
if __name__ == "__main__":
    # First, visualize a single trajectory
    print("Visualizing a single trajectory:")
    k_demo = 2
    steps = plot_single_trajectory(k_demo)
    print(f"Steps to escape for k={k_demo}: {steps}")
    
    # Analyze for different k values
    print("\nAnalyzing average escape time vs k:")
    k_values = [1, 2, 3, 4, 5, 6, 7, 8]
    avg_steps = analyze_escape_time(k_values, num_walks=300)
    
    # Theoretical prediction
    theoretical = theoretical_analysis(k_values)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, avg_steps, 'bo-', label='Simulation', linewidth=2, markersize=8)
    plt.plot(k_values, theoretical, 'r--', label=r'Theoretical $\sim k^2$', linewidth=2)
    
    plt.xlabel('k')
    plt.ylabel('Average Steps to Escape')
    plt.title('Escape Time from Square Box vs k')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Use log scale to see power law behavior
    plt.show()
    
    # Also plot linear scale with quadratic fit
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, avg_steps, 'bo-', label='Simulation', linewidth=2, markersize=8)
    
    # Quadratic fit
    k_fit = np.array(k_values)
    steps_fit = np.array(avg_steps)
    quadratic_fit = np.polyfit(k_fit**2, steps_fit, 1)
    fit_curve = quadratic_fit[0] * k_fit**2 + quadratic_fit[1]
    plt.plot(k_fit, fit_curve, 'r--', label=f'Quadratic fit: {quadratic_fit[0]:.2f}k² + {quadratic_fit[1]:.2f}', linewidth=2)
    
    plt.xlabel('k')
    plt.ylabel('Average Steps to Escape')
    plt.title('Escape Time from Square Box (Quadratic Scaling)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"\nQuadratic fit parameters: τ = {quadratic_fit[0]:.2f}k² + {quadratic_fit[1]:.2f}")
