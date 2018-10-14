import numpy as np
import sys
import matplotlib.pyplot as plt


def plot_residual(
	res, 
	plot_name='Residuos', 
	title='Residuos'):
	print("Plotando")
	plt.close("all")
	plt.hist(res)
	plt.title(title)
	plt.savefig(plot_name)