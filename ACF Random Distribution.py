import numpy as np
import matplotlib.pyplot as plt


# Galaxies not taking into account redshift
# number of galaxies
num_galaxies = 10000

# randomly generated mass, 2D position (right ascension and declination), velocity, and redshift of galaxies
galaxy_mass = np.random.uniform(low=1e9, high=1e12, size=num_galaxies)
galaxy_ascension = np.random.uniform(low=34.6, high=34.7, size=num_galaxies)
galaxy_declination = np.random.uniform(low=-5.5, high=-5.3, size=num_galaxies)
galaxy_redshift = np.random.uniform(low=0.5, high=3, size=num_galaxies)

# combine mass, 2D position, velocity, and redshift into a single array
galaxies = np.column_stack((galaxy_mass, galaxy_ascension, galaxy_declination, galaxy_redshift))

#%%
"""This is checking that this plots that is correctly"""
plt.figure()
plt.plot(galaxy_ascension, galaxy_declination, 'b.')
plt.show()

#%%
"""Trying to filter out the galaxies at different redshift intervals"""
RA_min, RA_max = 34.6, 34.7
Dec_min, Dec_max = -5.5, -5.3

# number of galaxies
num_galaxies = 10000# maximum and minimum redshifts
max_redshift = 3
min_redshift = 0.5# probability distribution for redshift
ID = np.arange(1,10001,1)
redshift_prob = np.linspace(max_redshift, min_redshift, num_galaxies)
redshift_prob = redshift_prob / redshift_prob.sum()# randomly generated redshift of galaxies
galaxy_redshift = np.random.choice(np.linspace(min_redshift, max_redshift, num_galaxies), size=num_galaxies, p=redshift_prob)# randomly generated mass, 2D position (right ascension and declination), velocity, and redshift of galaxies
galaxy_mass = np.random.uniform(low=1e9, high=1e12, size=num_galaxies)
galaxy_ascension = np.random.uniform(low=34.6, high=34.7, size=num_galaxies)
galaxy_declination = np.random.uniform(low=-5.5, high=-5.3, size=num_galaxies)# combine mass, 2D position, velocity, and redshift into a single array
galaxies = np.column_stack((galaxy_mass, galaxy_ascension, galaxy_declination, galaxy_redshift))

plt.plot(ID, galaxy_redshift, 'k.')
plt.show()


#%% 

"""
CODING THE ACF
"""

""""I found this on useblackbok.io/search , but need to """
def estimated_autocorrelation(x):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

"""
1. Set the centre of the box to be the centre of the analysis
2. Decide on the size of the theta bins
3. For each separation bin
    - Calculate the distance between all of the data-data pairs
    - Calculate the distance between all random-random pairs
    - Calulcate the distance between all data-random pairs
4. Input results into the L&S estimator
"""

RA_centre = (RA_min + RA_max) / 2
Dec_centre = (Dec_min + Dec_max) / 2


"""
UNCOMMENT WHEN ALL PREVIOUS STEPS HAVE BEEN CODED
LS_est = (DD-2*DR+RR)/RR
"""

# %%
