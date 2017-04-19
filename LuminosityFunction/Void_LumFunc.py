# Luminosity Function of Void Glaxies using SDSS Data
# Written by S.Aliei
# saeidaliei@gmail.com

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.style.use('ggplot')

textfont = {'fontname':'monospace', 'fontsize':10}
axisfont = {'fontname': 'serif', 'fontsize': 12}

# initial values for Schechter function
init_ScheParams=[-2, -25, 3]

# definition of schechter function by absolute magnitude, M
def ScheFunc(M, alpha, Mstar, Phistar):
	return (0.4*np.log(10)) * (Phistar) * (10**(0.4*(alpha+1)*(Mstar-M))) * (np.exp(-10**(0.4*(Mstar-M))))

# loading data
id_SDSS_DR13, RA, DEC, redshift, abs_mag_R, id_void, radius_void,density_contrast_void = np.loadtxt("void_galaxies.txt", unpack=True)
M = abs_mag_R
# choosing optimal bin sizes, a challenge!
# different bin size estimators:
# for discussions reffer to:
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
# bin_est = ['auto', 'fd', 'doane', 'scott', 'rice', 'sturges', 'sqrt', 100, 10]
# optimum bin obtained from trial and error:
bin_est = 9
hist_plot = plt.hist(M, bins=bin_est, log=True, alpha=0.6)
plt.xlabel(r'$\mathcal{M}_r$',**axisfont)
plt.ylabel(r'Log $\mathcal{\Phi}(\mathcal{M}_r)$',**axisfont)

# number of galaxies in each bin
num_in_bins = hist_plot[0]
# bin intervals
bin_edges = hist_plot[1]
# mid point of each bin
bins_mid_point = (bin_edges[1:]+bin_edges[:-1]) / 2
# poisson error for each bin
sigma = np.sqrt(num_in_bins)

# fitting points obtained above to Schecter Function,
# using least sqare methode(same as chi square method, but for non-Guassian distribution)
# we use the library provided by Scipy's Optimization module
# curve_fit, returns two sets of numpy array:
# popt: optimized(fitted) parameters
# pcov: covariance matrix for correlated errors of parameters
popt, pcov = curve_fit(ScheFunc, bins_mid_point, num_in_bins, 
	                                             sigma=sigma,
	                                             p0=init_ScheParams)
# errors in parameters from covariance matrix
perr = np.sqrt(np.diag(pcov))
# area integral under schechter function
integ = np.trapz(ScheFunc(bins_mid_point, popt[0], popt[1], popt[2]), bins_mid_point)
# normalized Phistar
Phistar_normalized = popt[2]/integ
# normalized Phistar error
Phistar_error_normalized = perr[2]/integ

print(70*'*')
print("Fitted Paramters:\n")
print("alpha: {}, alpha_error: {}" .format(popt[0], perr[0]))
print("Mstar: {}, Mstar_error: {}" .format(popt[1], perr[1]))
print("Phistar: {}, Phistar_error: {}" .format(Phistar_normalized, Phistar_error_normalized))
print("\narea under curve: {}" .format(integ))
print("\nCovariance Matrix: \n{}" .format(pcov))
print(70*'*')	

plt.plot(bins_mid_point, num_in_bins, '*', markersize=5)
plt.plot(bins_mid_point,ScheFunc(bins_mid_point, popt[0], popt[1], popt[2]),linestyle='-', linewidth=2)
plt.errorbar(bins_mid_point, num_in_bins, yerr=sigma, fmt='.')

alphalabel = r'-Best fit $\alpha$ = ' + str('%3.2f' % popt[0]) + r'$\pm$' + str('%3.2f' % perr[0])
Mstarlabel = r'-Best fit $\mathcal{M}^*$ = ' + str('%3.2f' % popt[1]) + r'$\pm$' + str('%3.2f' % perr[1])
Phistarlabel = r'-Best fit $\mathcal{\Phi}^*$ = ' + str('%3.2f' % Phistar_normalized) + r'$\pm$' + str('%3.2f' % Phistar_error_normalized)

plt.text(-21,6,alphalabel,**textfont)
plt.text(-21,5,Mstarlabel,**textfont)
plt.text(-21,4,Phistarlabel,**textfont)

plt.savefig('Void_LumFunc.pdf')
plt.show()