import numpy as np
import sys
import matplotlib.pyplot as plt


def plot_residual(
	res, plot_name='Resíduos', color='blue', title='Resíduos', show=False):
	plt.close("all")
	plt.figure(figsize=(8, 6))
	plt.hist(res, color=color)
	plt.title(title)
	plt.show()
	plt.savefig(plot_name)