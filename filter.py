import os
import sys
import time

import numpy as np
from astropy.io import fits as pyfits
from scipy.spatial import KDTree

class training_set():
	def __init__(self):
		self.real = None
		self.bogus = None
	
	def load(self):
		print '[%s]: Loading statinary catalog' %(time.strftime("%D %H:%M:%S"))
		sc = np.loadtxt('stationary_catalog.2')
		treesc = KDTree(sc)
		srclist = filter(lambda x: x.startswith('det.') and x.endswith('fits'), os.listdir('.'))
		
		srclist.sort()
		#srclist = srclist[:int(len(srclist)/5.0)]
		#srclist = srclist[:1]
		for i, srcn in enumerate(srclist):
			print '[%s]: Loading fits table: %03i/%03i' %(time.strftime("%D %H:%M:%S"), i+1, len(srclist))
			src = pyfits.open(srcn)
			#print src[1].data['flags']
			#break
			radec = np.degrees(src[1].data['coord'])
			ff = np.unpackbits(src[1].data['flags']).astype('bool').reshape((len(src[1].data), 64))
			#print ff[:,10]
			#print ff[:,36]
			#print ff[:,21]
			treesrc = KDTree(radec)
			match = treesrc.query_ball_tree(treesc,0.5/3600)
			l = [i for i, m in enumerate(match) if m==[]]
			scl = [i for i, m in enumerate(match) if m!=[]]
			ns = src[1].data[l]
			ns_flag = ff[l][:,10]*ff[l][:,36]+ff[l][:,21]
			#print l
			#print ns_flag
			if 'nonallsrc' in locals():
				nonallsrc = np.concatenate(( nonallsrc, ns[ns_flag]), axis=0)
				#nonallsrc = np.concatenate(( nonallsrc, ns), axis=0)
			else:
				nonallsrc = ns[ns_flag]
				#nonallsrc = ns
			if 'allsrc' in locals():
				allsrc = np.concatenate(( allsrc, src[1].data[scl]), axis=0)
			else:
				allsrc = src[1].data[scl]
		self.real = allsrc
		#print len(allsrc), len(nonallsrc), len(nonallsrc[nonallsrc['parent']!=0])
		#print nonallsrc['flags']
		self.bogus = nonallsrc#[nonallsrc['flags'][:, 10] * nonallsrc['flags'][:, 36] + nonallsrc['flags'][:, 21] ]#[nonallsrc['parent']!=0]
		np.savetxt('station.cat', np.degrees(allsrc['coord']))
		np.savetxt('nonstation.cat', np.degrees(nonallsrc['coord']))

def main():
	tset = training_set()
	tset.load()

if __name__ == '__main__':
	main()
	
	
