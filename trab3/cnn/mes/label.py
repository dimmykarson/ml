#!/usr/bin/python
import sys
import numpy as np


def main(filein):

	
	file = open(filein)
	file2 = open("test.txt", 'w')
	
	files = []
	labels = []
	i =0

	line = file.readline()
	while line:
		
		i = i+1
		t = line.split('.')
		fname = './labels/' + t[0] + '.inf' 
		
		f2 = open(fname)
		l = f2.readline()
		lb = l.split('|')
		print t[0], lb[0]
		f2.close()
		
				
		#file2.write(valores[i-1] + ' ' )
		#for j in range(0,i-1):
		file2.write(t[0] + '.tif ' + l[0] + '\n')

		#file2.write("\n" )
		


		line = file.readline()

	file.close
	
	
	#for x in range(len(files)):
	#	file3.write(files[x] + " " + str(labels[x]) + "\n")
		
	
	
	
	file2.close

if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Usage: tolibsvm.py <lista de arquivos> <quantidade no treinamento>")
	main(sys.argv[1])

