import os
import sys
import time

import numpy as np
import pyfits
from scipy.spatial import KDTree

class stat_ns:
        def __init__(self, fits_table, stationary):
                self.fits_table = fits_table
                self.stationary = stationary
                self.real = None
                self.bogus = None

        def load(self):
                #print '[%s]: Loading statinary catalog' %(time.strftime("%D %H:%M:%S"))
                #sc = np.loadtxt(self.stationary)
                sc = self.stationary
                treesc = KDTree(sc)
                srclist = [self.fits_table]

                srclist.sort()

                for i, srcn in enumerate(srclist):
                        print '[%s]: Loading fits table: %03i/%03i' %(time.strftime("%D %H:%M:%S"), i+1, len(srclist))
                        src = pyfits.open(srcn)
                        radec = np.degrees(src[1].data['coord'])
                        ff = np.unpackbits(src[1].data['flags']).astype('bool').reshape((len(src[1].data), 64))
                        treesrc = KDTree(radec)
                        match = treesrc.query_ball_tree(treesc,0.5/3600)
                        l = [i for i, m in enumerate(match) if m==[]]
                        scl = [i for i, m in enumerate(match) if m!=[]]
                        ns = src[1].data[l]
                        ns_flag = ~(ff[l][:,10]*ff[l][:,36]+ff[l][:,21])
                        ns_flag = ns['parent'] == 1
			if 'nonallsrc' in locals():
                                #nonallsrc = np.concatenate(( nonallsrc, ns[ns_flag]), axis=0)
                        	nonallsrc = np.concatenate(( nonallsrc, ns), axis=0)
			else:   
                                #nonallsrc = ns[ns_flag]
				nonallsrc = ns
                        if 'allsrc' in locals():
                                allsrc = np.concatenate(( allsrc, src[1].data[scl]), axis=0)
                        else:   
                                allsrc = src[1].data[scl]
                self.stat = allsrc
                self.ns = nonallsrc
                np.savetxt('station.cat', np.degrees(allsrc['coord']))
                np.savetxt('nonstation.cat', np.degrees(nonallsrc['coord']))

def main():
        tset = stat_ns('det.0072556.06505.fits',np.loadtxt('stationary_catalog.2_206.7_+000.0_06505'))
        tset.load()

if __name__ == '__main__':
        main()

