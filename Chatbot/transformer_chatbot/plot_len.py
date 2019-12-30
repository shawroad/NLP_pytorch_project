"""

@file   : plot_len.py

@author : xiaolu

@time   : 2019-12-26

"""
import matplotlib.pyplot as plt
import numpy as np


x = np.load('./data/lengths.npy')

# Fixing random state for reproducibility
np.random.seed(19680801)

# mu, sigma = 100, 15
# x = mu + sigma * np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(x, 10, density=True, facecolor='g', alpha=0.75)


plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.xlim(40, 160)
plt.ylim(0, 0.03)
plt.grid(True)
plt.show()