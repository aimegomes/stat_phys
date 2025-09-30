import numpy as np
import matplotlib.pyplot as plt

### GERANDO ARRAY ALEATÓRIO
N_values = [10, 100, 1000, 10000]  # Number of elements in array
p = 0.5  # probability
x = 50000  # repetitions

mean_M_per_N = []
mean_n_plus_per_N = []
std_M_per_N = []
std_n_plus_per_N = []

for N in N_values:
    M_values = np.zeros(x)
    n_plus_values = np.zeros(x)
    n_minus_values = np.zeros(x)

    print(f"*****INITIATING LOOP FOR N={N}*****")

    for i in range(x):
        # Generate random array
        values_array = np.random.choice([-1, 1], size=N, p=[1-p, p])
        # Calculate M and counts
        M_values[i] = np.sum(values_array)  # M = n_plus - n_minus
        n_plus_values[i] = (M_values[i] + N) / 2
        n_minus_values[i] = N - n_plus_values[i]

    print("*****LOOP FINISHED*****")
    print("- M VALUES OBTAINED:", M_values)

    print("*****START OF STATISTIC CALCULATIONS*****")
    # STATS FOR THIS ITERATION
    mean_M = np.mean(M_values)
    mean_n_plus = np.mean(n_plus_values)
    std_M = np.std(M_values)
    std_n_plus = np.std(n_plus_values)
    var_M = np.var(M_values)
    var_n_plus = np.var(n_plus_values)

    theo_value_n_plus = p*N
    theo_value_M = (2*p-1)*N #expected mean value of M
    std_div_exp_n_plus= std_n_plus/theo_value_n_plus_v
    # MEAN VALUES FOR FINAL GRAPH
    mean_M_per_N.append(mean_M)
    mean_n_plus_per_N.append(mean_n_plus)
    std_M_per_N.append(std_M)
    std_n_plus_per_N.append(std_n_plus)

    print(f"Mean M for N={N}: {mean_M:.2f}")
    print(f"Mean n_plus for N={N}: {mean_n_plus:.2f}")
    print(f"Standard deviation M for N={N}: {std_M:.2f}")
    print(f"Standard deviation n_plus for N={N}: {std_n_plus:.2f}")
    print(f"Var M for N={N}: {var_M:.2f}")
    print(f"Var n_plus for N={N}: {var_n_plus:.2f}")
    #print(f"Expected value M for N={N}: {var_M:.2f}"}
    print(f"Expected value n_plus for N={N}: {exp_value}")

    print(f"Ratio STD/Expected value n_plus for N={N}: {std_div_exp}")

    print("*****HISTOGRAMS*****")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Histogram for M
    ax1.hist(M_values, bins=20, alpha=0.7, color='navy', edgecolor='black')
    ax1.axvline(mean_M, color='red', linestyle='--', label=f'Mean: {mean_M:.2f}')
    ax1.set_title(f'Distribution of M (N={N}, p={p})')
    ax1.set_xlabel('M value')
    ax1.set_ylabel('Frequency')
    ax1.legend()

    # Histogram for n_plus
    ax2.hist(n_plus_values, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(mean_n_plus, color='red', linestyle='--', label=f'Mean: {mean_n_plus:.2f}')
    ax2.set_title(f'Distribution of N+ (N={N}, p={p})')
    ax2.set_xlabel('N+ value')
    ax2.set_ylabel('Frequency')
    ax2.legend()

    plt.tight_layout()
    plt.show()

# FINAL PLOT: Mean M vs N
print("\n*****FINAL RESULTS: MEAN M VS N*****")
for i, N in enumerate(N_values):
    print(f"N={N}: Mean M = {mean_M_per_N[i]:.2f}")
