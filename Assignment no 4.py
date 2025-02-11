import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
n_values = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]

for n in n_values:
    dice1 = np.random.randint(1, 7, n)
    dice2 = np.random.randint(1, 7, n)
    sums = dice1 + dice2
    h, h2 = np.histogram(sums, range(2, 14))
    plt.bar(h2[:-1], h / n, width=0.8, alpha=0.5, color='blue' )
    plt.title(f"Histogram of Dice Sums for n={n}")
    plt.xlabel("Sum of Dice")
    plt.ylabel("Frequency")
    plt.show()
# Observations
print("As n increases, the histogram approaches the theoretical probabilities for the sum of two dice.")
print("This reflects regression to the mean, as the observed frequencies converge to expected values with larger samples.")
