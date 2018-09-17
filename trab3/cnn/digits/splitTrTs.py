#!/usr/bin/python
import sys
import time
import numpy as np
import random

np.random.seed(seed=int(time.time()))

def main(filein, sizeTr):

	
	file = open(filein)
	file2 = open("train.txt", 'w')
	file3 = open("test.txt", 'w')
	
	files = []
	labels = []
	i =0

	line = file.readline()
	while line:
		
		i = i+1
		t = line.split()
		fname = t[0].split('/')
		label = t[1]
		
		print fname[1], label
		files.append(fname[1])
		labels.append(label)
		
		#file2.write(valores[i-1] + ' ' )
		#for j in range(0,i-1):
		#   file2.write(str(j+1) + ":" + valores[j] + " ")

		#file2.write("\n" )
		


		line = file.readline()

	file.close
	
	## selects part of the data for training and the remaining for testing.
	print '------'
	removed = 1
	for x in range(int(sizeTr)):
		ind =  random.randint(0,i-removed)	
		print files[ind], labels[ind], ind
		file2.write(files[ind] + " " + str(labels[ind]) + "\n")
		files.pop(ind)
		labels.pop(ind)
		removed = removed + 1
		

	### testing
	print "for testing: ", len(files)

	for x in range(len(files)):
		file3.write(files[x] + " " + str(labels[x]) + "\n")
		
	
	
	
	file2.close
	file3.close

if __name__ == "__main__":
	if len(sys.argv) != 3:
		sys.exit("Usage: tolibsvm.py <lista de arquivos> <quantidade no treinamento>")
	main(sys.argv[1], sys.argv[2])

