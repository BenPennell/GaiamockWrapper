# Ben's gaiamock wrapper and tools and such

Ben Pennell

--
_____

directory:

`Forward.py` has an object-oriented approach for calculating likelihoods with astrometric signals for catalogues of objects or single objects

`GaiamockWrapper.py` is a wrapper for `gaiamock` that is streamlined to marginalise over parameters and rapidly calculate RUWE and types of orbital solutions

helpful (hopefully) notebooks:

`FastRuweExample.ipynb` briefly goes over how to calculate RUWE using `GaiamockWrapper.py` and the speedups I've used 

`gwexamples.ipynb` has some examples for marginalisation using `GaiamockWrapper.py`

`mcmcexamples.ipynb` has an example for using mcmc to use `GaiamockWrapper.py` repeatedly to constrain parameters for single objects

documentation is not at all complete. I'll answer questions and add features as requested.