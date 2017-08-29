from numpy import *
from filter_single import stat_ns
from sklearn.externals import joblib
import pyfits

class ns_for_ml:
	def __init__(self, ns):
		self.ns = ns
		self.nsClass = None
		self.feature_ns = None
		self.extract_feature()
		self.extract_coord()
	
	def denan(self, data):
		data[isnan(data)] = -999
	
	def extract_feature(self):
		ff = unpackbits(self.ns[:]['flags']).reshape((len(self.ns[:]), 64))
		ns_list = [self.ns[:]['parent'], self.ns[:]['shape_hsm_moments'][:,0], 
				self.ns[:]['shape_hsm_moments'][:,1], self.ns[:]['shape_hsm_moments'][:,2],
				self.ns[:]['shape_hsm_psfMoments'][:,0], self.ns[:]['shape_hsm_psfMoments'][:,1],
				self.ns[:]['shape_hsm_psfMoments'][:,2],
				self.ns[:]['shape_sdss'][:,0], self.ns[:]['shape_sdss'][:,1], self.ns[:]['shape_sdss'][:,2],
				self.ns[:]['shape_sdss_psf'][:,0], self.ns[:]['shape_sdss_psf'][:,1], self.ns[:]['shape_sdss_psf'][:,2],
				self.ns[:]['flux_aperture'][:,0], self.ns[:]['flux_aperture'][:,1], 
				self.ns[:]['flux_aperture'][:,2], self.ns[:]['flux_aperture'][:,3],
				self.ns[:]['flux_aperture'][:,4], self.ns[:]['flux_aperture'][:,5],
				self.ns[:]['flux_aperture_err'][:,0], self.ns[:]['flux_aperture_err'][:,1],
				self.ns[:]['flux_aperture_err'][:,2], self.ns[:]['flux_aperture_err'][:,3],
				self.ns[:]['flux_aperture_err'][:,4], self.ns[:]['flux_aperture_err'][:,5],
				self.ns[:]['flux_aperture_nInterpolatedPixel'][:,0], self.ns[:]['flux_aperture_nInterpolatedPixel'][:,1],
                                self.ns[:]['flux_aperture_nInterpolatedPixel'][:,2], self.ns[:]['flux_aperture_nInterpolatedPixel'][:,3],
                                self.ns[:]['flux_aperture_nInterpolatedPixel'][:,4], self.ns[:]['flux_aperture_nInterpolatedPixel'][:,5],
				self.ns[:]['flux_gaussian'],
				self.ns[:]['flux_gaussian_err'], self.ns[:]['flux_naive'], self.ns[:]['flux_naive_err'], 
				self.ns[:]['flux_kron'], self.ns[:]['flux_kron_err'], self.ns[:]['flux_kron_radius'],
				self.ns[:]['flux_kron_psfRadius'], self.ns[:]['flux_psf'], self.ns[:]['flux_psf_err'],
				self.ns[:]['flux_sinc'], self.ns[:]['flux_sinc_err'], self.ns[:]['flux_gaussian_apcorr'], 
				self.ns[:]['flux_gaussian_apcorr_err'], self.ns[:]['flux_kron_apcorr'], self.ns[:]['flux_kron_apcorr_err'],
				self.ns[:]['flux_psf_apcorr'], self.ns[:]['flux_psf_apcorr_err']]
		
		map(self.denan, ns_list)
		self.feature_ns = array(ns_list).T
		#print "stationary sources: {0}".format(len(self.feature_sat))
		#print "non-stationary sources: {0}".format(len(self.feature_ns))
		
	def extract_coord(self):
		self.ns_coord = degrees(self.ns[:]['coord'])
		
	
	
def main():
	stationary = 'stationary_catalog.2_206.7_+000.0_06505'
	fits_tab = 'det.0072556.06505.fits'
	mlns_filename = '{}.mlns'.format(fits_tab)
	RFModel = '/sciproc/disk3/charles/HSC_healpix/208.1_-001.2_06499/RFmod_1216/real_bogus_RF_run0.pkl'
	#print 'load ML modle: {}'.format(RFModel)
	RFmod = joblib.load(RFModel)
	#print 'load fits table: {}'.format(fits_tab)
	sc = loadtxt(stationary)
	data = stat_ns(fits_tab, sc)
	data.load()
	ns_before_ml = ns_for_ml(data.ns)
	ns_before_ml.nsClass = RFmod.predict(ns_before_ml.feature_ns)
	mlns_mask = ns_before_ml.nsClass == 'real'
	mlns_bogus = ns_before_ml.nsClass == 'bogus'
	mlns = data.ns[mlns_mask]
	bogus = data.ns[mlns_bogus]
	for i in mlns['coord']:
		print i[0]*180./pi, i[1]*180./pi
	#print 'write mlns: {}'.format(mlns_filename)
	pyfits.writeto(mlns_filename, mlns, clobber=True)
		
	
if __name__ == '__main__':
	main()
	
