from numpy import *
from sklearn import *
from scipy.spatial import KDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, roc_curve
from filter import training_set
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import pickle
import os
from matplotlib import pyplot as plt

class load_tset:
	def __init__(self, real, bogus):
		self.real_array = real
		self.bogus_array = bogus
		self.Class = []
		self.feature = None
		self.feature_real = None
		self.feature_bogus = None
		self.real_coord = None
		self.bogus_coord = None
		self.ext_list = None
		self.extract_feature()
		self.extract_coord()
		
	def load_ext_list(self):
		coord_ext = loadtxt(self.ext_list)
		feature_new_bogus = []
		raw_tree = KDTree(self.bogus_coord)
		ext_tree = KDTree(coord_ext)
		match = ext_tree.query_ball_tree(raw_tree, 0.01/3600.)
		for i in match:
			if i != []:
				feature_new_bogus.append(self.feature_bogus[i[0]])
		
		self.feature_bogus = feature_new_bogus
			
	def denan(self, data):
		data[isnan(data)] = -999
	
	def extract_feature(self):
		#self.real_array[:]['parent'], 
		#self.real_array[:]['shape_hsm_regauss_e1'], self.real_array[:]['shape_hsm_regauss_e2'],
		#		self.real_array[:]['shape_hsm_regauss_sigma'], self.real_array[:]['shape_hsm_regauss_resolution'],
		ff = unpackbits(self.real_array[:]['flags']).reshape((len(self.real_array[:]), 64))
		#print ff[:,10]
		#print self.real_array[:]['parent']
		real_list = [self.real_array[:]['parent'], self.real_array[:]['shape_hsm_moments'][:,0], 
				self.real_array[:]['shape_hsm_moments'][:,1], self.real_array[:]['shape_hsm_moments'][:,2],
				self.real_array[:]['shape_hsm_psfMoments'][:,0], self.real_array[:]['shape_hsm_psfMoments'][:,1],
				self.real_array[:]['shape_hsm_psfMoments'][:,2],
				self.real_array[:]['shape_sdss'][:,0], self.real_array[:]['shape_sdss'][:,1], self.real_array[:]['shape_sdss'][:,2],
				self.real_array[:]['shape_sdss_psf'][:,0], self.real_array[:]['shape_sdss_psf'][:,1], self.real_array[:]['shape_sdss_psf'][:,2],
				self.real_array[:]['flux_aperture'][:,0], self.real_array[:]['flux_aperture'][:,1], 
				self.real_array[:]['flux_aperture'][:,2], self.real_array[:]['flux_aperture'][:,3],
				self.real_array[:]['flux_aperture'][:,4], self.real_array[:]['flux_aperture'][:,5],
				self.real_array[:]['flux_aperture_err'][:,0], self.real_array[:]['flux_aperture_err'][:,1],
				self.real_array[:]['flux_aperture_err'][:,2], self.real_array[:]['flux_aperture_err'][:,3],
				self.real_array[:]['flux_aperture_err'][:,4], self.real_array[:]['flux_aperture_err'][:,5],
				self.real_array[:]['flux_aperture_nInterpolatedPixel'][:,0], self.real_array[:]['flux_aperture_nInterpolatedPixel'][:,1],
                                self.real_array[:]['flux_aperture_nInterpolatedPixel'][:,2], self.real_array[:]['flux_aperture_nInterpolatedPixel'][:,3],
                                self.real_array[:]['flux_aperture_nInterpolatedPixel'][:,4], self.real_array[:]['flux_aperture_nInterpolatedPixel'][:,5],
				self.real_array[:]['flux_gaussian'],
				self.real_array[:]['flux_gaussian_err'], self.real_array[:]['flux_naive'], self.real_array[:]['flux_naive_err'], 
				self.real_array[:]['flux_kron'], self.real_array[:]['flux_kron_err'], self.real_array[:]['flux_kron_radius'],
				self.real_array[:]['flux_kron_psfRadius'], self.real_array[:]['flux_psf'], self.real_array[:]['flux_psf_err'],
				self.real_array[:]['flux_sinc'], self.real_array[:]['flux_sinc_err'], self.real_array[:]['flux_gaussian_apcorr'], 
				self.real_array[:]['flux_gaussian_apcorr_err'], self.real_array[:]['flux_kron_apcorr'], self.real_array[:]['flux_kron_apcorr_err'],
				self.real_array[:]['flux_psf_apcorr'], self.real_array[:]['flux_psf_apcorr_err']]
		map(self.denan, real_list)
		size = len(array(real_list).T) 
		flag = random.rand(size) < 0.025
		self.feature_real = array(real_list).T[flag]
		#self.bogus_array[:]['parent'],
		#self.bogus_array[:]['shape_hsm_regauss_e1'], self.bogus_array[:]['shape_hsm_regauss_e2'],
		#		self.bogus_array[:]['shape_hsm_regauss_sigma'], self.bogus_array[:]['shape_hsm_regauss_resolution'],
		ff_b = unpackbits(self.bogus_array[:]['flags']).reshape((len(self.bogus_array), 64))
		bogus_list = [self.bogus_array[:]['parent'], self.bogus_array[:]['shape_hsm_moments'][:,0], 
				self.bogus_array[:]['shape_hsm_moments'][:,1], self.bogus_array[:]['shape_hsm_moments'][:,2],
				self.bogus_array[:]['shape_hsm_psfMoments'][:,0], self.bogus_array[:]['shape_hsm_psfMoments'][:,1],
				self.bogus_array[:]['shape_hsm_psfMoments'][:,2],
				self.bogus_array[:]['shape_sdss'][:,0], self.bogus_array[:]['shape_sdss'][:,1], self.bogus_array[:]['shape_sdss'][:,2],
				self.bogus_array[:]['shape_sdss_psf'][:,0], self.bogus_array[:]['shape_sdss_psf'][:,1], self.bogus_array[:]['shape_sdss_psf'][:,2],
				self.bogus_array[:]['flux_aperture'][:,0], self.bogus_array[:]['flux_aperture'][:,1], 
				self.bogus_array[:]['flux_aperture'][:,2], self.bogus_array[:]['flux_aperture'][:,3],
				self.bogus_array[:]['flux_aperture'][:,4], self.bogus_array[:]['flux_aperture'][:,5],
				self.bogus_array[:]['flux_aperture_err'][:,0], self.bogus_array[:]['flux_aperture_err'][:,1],
				self.bogus_array[:]['flux_aperture_err'][:,2], self.bogus_array[:]['flux_aperture_err'][:,3],
				self.bogus_array[:]['flux_aperture_err'][:,4], self.bogus_array[:]['flux_aperture_err'][:,5],
				self.bogus_array[:]['flux_aperture_nInterpolatedPixel'][:,0], self.bogus_array[:]['flux_aperture_nInterpolatedPixel'][:,1],
                                self.bogus_array[:]['flux_aperture_nInterpolatedPixel'][:,2], self.bogus_array[:]['flux_aperture_nInterpolatedPixel'][:,3],
                                self.bogus_array[:]['flux_aperture_nInterpolatedPixel'][:,4], self.bogus_array[:]['flux_aperture_nInterpolatedPixel'][:,5],
				self.bogus_array[:]['flux_gaussian'],
				self.bogus_array[:]['flux_gaussian_err'], self.bogus_array[:]['flux_naive'], self.bogus_array[:]['flux_naive_err'], 
				self.bogus_array[:]['flux_kron'], self.bogus_array[:]['flux_kron_err'], self.bogus_array[:]['flux_kron_radius'],
				self.bogus_array[:]['flux_kron_psfRadius'], self.bogus_array[:]['flux_psf'], self.bogus_array[:]['flux_psf_err'],
				self.bogus_array[:]['flux_sinc'], self.bogus_array[:]['flux_sinc_err'], self.bogus_array[:]['flux_gaussian_apcorr'], 
				self.bogus_array[:]['flux_gaussian_apcorr_err'], self.bogus_array[:]['flux_kron_apcorr'], self.bogus_array[:]['flux_kron_apcorr_err'],
				self.bogus_array[:]['flux_psf_apcorr'], self.bogus_array[:]['flux_psf_apcorr_err']]
		map(self.denan, bogus_list)
		self.feature_bogus = array(bogus_list).T
		#print bogus_list
	def make_tset(self):
		self.feature = array(list(self.feature_real)+list(self.feature_bogus))
		[self.Class.append('real') for i in range(len(self.feature_real))]	
		[self.Class.append('bogus') for i in range(len(self.feature_bogus))]
		print "real sources: {0}".format(len(self.feature_real))
		print "bogus sources: {0}".format(len(self.feature_bogus))

	def extract_coord(self):
		self.real_coord = degrees(self.real_array[:]['coord'])
		self.bogus_coord = degrees(self.bogus_array[:]['coord'])
		
def save_model(model,filename):
	joblib.dump(model, filename) 
	
	
def main():
	#tset_raw = training_set()
	#tset_raw.load()
	#tset = load_tset(tset_raw.real, tset_raw.bogus)
	#tset.ext_list = 'ns_bogus.list'
	#tset.load_ext_list()
	#tset.make_tset()
	#with open('all_data_balance.pkl', 'wb') as output:
	#	pickle.dump(tset, output)
	tset = []
	with open("all_data.pkl", "rb") as data:
		tset = pickle.load(data)
	tset.extract_feature()
	tset.extract_coord()
	tset.Class = []
	print "real sources: {0}".format(len(tset.feature_real))
	print "bogus sources: {0}".format(len(tset.feature_bogus))
	tset.make_tset()
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    					tset.feature, tset.Class, test_size=0.01, random_state=0)
	print "There are %s/%s objects in the training/test sets" %(len(y_train), len(y_test))
	RFmod = RandomForestClassifier(n_estimators = 160, max_features='sqrt', oob_score = True, n_jobs = 12)
	RFmod.fit(X_train, y_train)
	save_model(RFmod,'RF_balance.pkl')
	#for i in range(10):
	#	RFmod = RandomForestClassifier(n_estimators = 20*(i+1), max_features='sqrt', oob_score = True, n_jobs = 12)
        #        RFmod.fit(X_train, y_train)
        #        save_model(RFmod,'RF_{0}_sqrt.pkl'.format(20*(i+1)))
	#	RFmod = RandomForestClassifier(n_estimators = 20*(i+1), max_features='log2', oob_score = True, n_jobs = 12)
        #        RFmod.fit(X_train, y_train)
        #        save_model(RFmod,'RF_{0}_log2.pkl'.format(20*(i+1)))
	
	#RFmod_list = filter(lambda x: x.startswith('RF') and x.endswith('pkl'), os.listdir('.'))
	#RFmod_list = ['RF_160_sqrt.pkl']
	#for i in RFmod_list:
		#print i
	#	RFmod = joblib.load(i)
		#print "important features {0}".format(RFmod.feature_importances_)
		#print "Score {0}".format(1-RFmod.score(X_train, y_train))
		#y_cv_preds = cross_validation.cross_val_predict(RFmod, X_test, y_test)
		#cm = confusion_matrix(y_test, y_cv_preds)
		#normalized_cm = cm.astype('float')/cm.sum(axis = 1)[:,newaxis]
		#print normalized_cm
		#print RFmod.decision_path(X_train[:1000])
		#print RFmod.classes_
	with open("all_data.pkl", "rb") as data:
                tset = pickle.load(data)
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                                        tset.feature, tset.Class, test_size=0.4, random_state=0)
	score = RFmod.predict_proba(X_test)
	fpr, tpr, thresholds = roc_curve(y_test, score[:,1], pos_label='real')  
	print score
	for n,j in enumerate(fpr):
		print j, tpr[n], thresholds[n]
		plt.plot(fpr, tpr)
		plt.savefig('test.png')
		#y_cv_preds = cross_validation.cross_val_predict(RFmod, X_test, y_test)
		#cm = confusion_matrix(y_test, y_cv_preds)
		#print cm
	
	#oob_error = 1-RFmod.score(X_train, y_train)
	#print "Out of Bag error is {:.1f}%".format(100*oob_error)
		#cv_accuracy = cross_validation.cross_val_score(RFmod, tset.feature, tset.Class, cv=10)
	#cv_error = 1-cv_accuracy.mean()
	#print 'The cross-validation error is {:.1f}%'.format(100*cv_error)
	#save_model(RFmod)
	#y_cv_preds = cross_validation.cross_val_predict(RFmod, X_test, y_test)
	#cm = confusion_matrix(y_test, y_cv_preds)
	#print cm
	#normalized_cm = cm.astype('float')/cm.sum(axis = 1)[:,newaxis]
	#print normalized_cm
	#plt.imshow(normalized_cm, interpolation = 'nearest')
	#plt.colorbar()
	#tick_marks = arange(2)
	#plt.xticks(tick_marks, ['bogus', 'real'], rotation=45)
	#plt.yticks(tick_marks, ['bogus', 'real'])
	#plt.tight_layout()
	#plt.show()
	#sample.Class = RFmod.predict(sample.feature)
	
if __name__ == '__main__':
	main()
	
