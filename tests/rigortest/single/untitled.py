import numpy as np
# for i in ['h10.npy', 'h3.npy',  'log.npy',  'sqrt.npy']:
# 	a = np.load(i)
# 	b = a[:100*1000]
# 	np.save(i,b)

for i in ['h10.npy', 'h3.npy',  'log.npy',  'sqrt.npy']:
	a = np.load(i)
	print(len(a)/(1000*100))