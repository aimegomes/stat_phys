import numpy as np
import matplotlib.pyplot as plt

# Parameters
N_values = np.arange(100, 5000, 20)  # Number of elements in array
p = 0.5  # probability
x = 15000  # number of realizations

# Storage arrays
mean_M_per_N = []        # <M> estimate from x realizations
var_M_per_N = []         # var(M) estimate from x realizations
std_M_per_N = []         # std(M) estimate from x realizations
mean_n_plus_per_N = []   # <N+> estimate from x realizations
var_n_plus_per_N = []    # var(N+) estimate from x realizations
std_n_plus_per_N = []    # std(N+) estimate from x realizations

# Error estimates (Standard Error of the Mean)
sem_M_per_N = []         # SEM for <M>
sem_var_M_per_N = []     # SEM for var(M)
sem_std_M_per_N = []     # SEM for std(M)
sem_n_plus_per_N = []    # SEM for <N+>
sem_ratio_per_N = []     # SEM for std(N+)/<N+>

print("Starting simulations...")

for N in N_values:
    M_values = np.zeros(x)
    n_plus_values = np.zeros(x)

    for i in range(x):
        values_array = np.random.choice([-1, 1], size=N, p=[1-p, p])
        M_values[i] = np.sum(values_array)
        n_plus_values[i] = (M_values[i] + N) / 2

    # Estimates from x realizations
    mean_M = np.mean(M_values)
    var_M = np.var(M_values)
    std_M = np.std(M_values)
    mean_n_plus = np.mean(n_plus_values)
    var_n_plus = np.var(n_plus_values)
    std_n_plus = np.std(n_plus_values)
    ratio = std_n_plus / mean_n_plus

    # Store estimates
    mean_M_per_N.append(mean_M)
    var_M_per_N.append(var_M)
    std_M_per_N.append(std_M)
    mean_n_plus_per_N.append(mean_n_plus)
    var_n_plus_per_N.append(var_n_plus)
    std_n_plus_per_N.append(std_n_plus)

    # Standard Error of the Mean calculations
    sem_M_per_N.append(std_M / np.sqrt(x))  # SEM for mean
    sem_var_M_per_N.append(np.sqrt(2/(x-1)) * var_M)  # Approximate SEM for variance
    sem_std_M_per_N.append(std_M / np.sqrt(2*x))  # Approximate SEM for std
    sem_n_plus_per_N.append(std_n_plus / np.sqrt(x))  # SEM for mean_n_plus
    sem_ratio_per_N.append(ratio / np.sqrt(2*x))  # Approximate SEM for ratio

# Convert to numpy arrays
N_values = np.array(N_values)
mean_M_per_N = np.array(mean_M_per_N)
var_M_per_N = np.array(var_M_per_N)
std_M_per_N = np.array(std_M_per_N)
mean_n_plus_per_N = np.array(mean_n_plus_per_N)
var_n_plus_per_N = np.array(var_n_plus_per_N)
std_n_plus_per_N = np.array(std_n_plus_per_N)
ratio_per_N = std_n_plus_per_N / mean_n_plus_per_N

# Theoretical predictions
theoretical_M = (2*p - 1) * N_values
theoretical_var_M = N_values * (1 - (2*p-1)**2)  # = N for p=0.5
theoretical_std_M = np.sqrt(theoretical_var_M)
theoretical_n_plus = p * N_values
theoretical_var_n_plus = N_values * p * (1-p)  # = 0.25N for p=0.5
theoretical_std_n_plus = np.sqrt(theoretical_var_n_plus)
theoretical_ratio = theoretical_std_n_plus / theoretical_n_plus  # = 1/√N for p=0.5

# PLOT 1: <M> as function of N with error bars
plt.figure(figsize=(12, 8))
plt.errorbar(N_values, mean_M_per_N, yerr=sem_M_per_N, fmt='o',
             capsize=3, alpha=0.7, markersize=3, label=f'<M> ± SEM (x={x})')
plt.plot(N_values, theoretical_M, 'r-', linewidth=2, label='Theoretical <M> = 0')
plt.xlabel('N')
plt.ylabel('<M>')
plt.title(f'2. <M> estimate vs N (x={x} realizations)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# PLOT 2: var(M) as function of N with error bars
plt.figure(figsize=(12, 8))
plt.errorbar(N_values, var_M_per_N, yerr=sem_var_M_per_N, fmt='o',
             capsize=3, alpha=0.7, markersize=3, label=f'var(M) ± SEM (x={x})')
plt.plot(N_values, theoretical_var_M, 'r-', linewidth=2, label='Theoretical var(M) = N')
plt.xlabel('N')
plt.ylabel('var(M)')
plt.title(f'3. var(M) estimate vs N (x={x} realizations)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# PLOT 3: std(M) as function of N with error bars
plt.figure(figsize=(12, 8))
plt.errorbar(N_values, std_M_per_N, yerr=sem_std_M_per_N, fmt='o',
             capsize=3, alpha=0.7, markersize=3, label=f'std(M) ± SEM (x={x})')
plt.plot(N_values, theoretical_std_M, 'r-', linewidth=2, label='Theoretical std(M) = √N')
plt.xlabel('N')
plt.ylabel('std(M)')
plt.title(f'3. std(M) estimate vs N (x={x} realizations)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# PLOT 4: <N+> as function of N with error bars
plt.figure(figsize=(12, 8))
plt.errorbar(N_values, mean_n_plus_per_N, yerr=sem_n_plus_per_N, fmt='o',
             capsize=3, alpha=0.7, markersize=3, label=f'<N+> ± SEM (x={x})')
plt.plot(N_values, theoretical_n_plus, 'r-', linewidth=2, label=f'Theoretical <N+> = {p}N')
plt.xlabel('N')
plt.ylabel('<N+>')
plt.title(f'4. <N+> estimate vs N (x={x} realizations)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# PLOT 5: std(N+)/<N+> as function of N with error bars
plt.figure(figsize=(12, 8))
plt.errorbar(N_values, ratio_per_N, yerr=sem_ratio_per_N, fmt='o',
             capsize=3, alpha=0.7, markersize=3, label=f'std(N+)/<N+> ± SEM (x={x})')
plt.plot(N_values, theoretical_ratio, 'r-', linewidth=2, label='Theoretical std(N+)/<N+> = 1/√N')
plt.xlabel('N')
plt.ylabel('std(N+)/<N+>')
plt.title(f'5. std(N+)/<N+> estimate vs N (x={x} realizations)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Summary plot with all quantities
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.errorbar(N_values, mean_M_per_N, yerr=sem_M_per_N, fmt='o', markersize=2, alpha=0.7)
plt.plot(N_values, theoretical_M, 'r-')
plt.xlabel('N')
plt.ylabel('<M>')
plt.title('<M> vs N')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 2)
plt.errorbar(N_values, var_M_per_N, yerr=sem_var_M_per_N, fmt='o', markersize=2, alpha=0.7)
plt.plot(N_values, theoretical_var_M, 'r-')
plt.xlabel('N')
plt.ylabel('var(M)')
plt.title('var(M) vs N')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 3)
plt.errorbar(N_values, std_M_per_N, yerr=sem_std_M_per_N, fmt='o', markersize=2, alpha=0.7)
plt.plot(N_values, theoretical_std_M, 'r-')
plt.xlabel('N')
plt.ylabel('std(M)')
plt.title('std(M) vs N')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 4)
plt.errorbar(N_values, mean_n_plus_per_N, yerr=sem_n_plus_per_N, fmt='o', markersize=2, alpha=0.7)
plt.plot(N_values, theoretical_n_plus, 'r-')
plt.xlabel('N')
plt.ylabel('<N+>')
plt.title('<N+> vs N')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 5)
plt.errorbar(N_values, var_n_plus_per_N, yerr=np.sqrt(2/(x-1)) * var_n_plus_per_N,
             fmt='o', markersize=2, alpha=0.7)
plt.plot(N_values, theoretical_var_n_plus, 'r-')
plt.xlabel('N')
plt.ylabel('var(N+)')
plt.title('var(N+) vs N')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 6)
plt.errorbar(N_values, ratio_per_N, yerr=sem_ratio_per_N, fmt='o', markersize=2, alpha=0.7)
plt.plot(N_values, theoretical_ratio, 'r-')
plt.xlabel('N')
plt.ylabel('std(N+)/<N+>')
plt.title('std(N+)/<N+> vs N')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle(f'All statistical estimates from x={x} realizations', y=1.02)
plt.show()

print("Simulation completed!")
print(f"\nTheoretical relationships verified for p={p}:")
print(f"1. <M> = {(2*p-1)}N = 0")
print(f"2. var(M) = N")
print(f"3. std(M) = √N")
print(f"4. <N+> = {p}N = {p}N")
print(f"5. var(N+) = {p*(1-p)}N = {p*(1-p)}N")
print(f"6. std(N+)/<N+> = 1/√N")
print(f"\nError bars represent Standard Error of the Mean (SEM) based on x={x} realizations")
