# Generating Mock Datasets

Ben Pennell, MPIA

February 20th, 2026
_____

Ignore everything in `./Legacy`, it's just here for completeness.

With `create_synthetic_data()` in `SyntheticData.py` you can input an astropy table of objects and make a mock catalogue that matches its distributions of ra,dec,pmra,pmdec,parallax,magnitude, and mass. Then you can assign companions to it based on a binary fraction and some choices of the period, mass ratio, and eccentricity distributions.

`SyntheticData.py` makes use of my random utility functions in `./utils/utils.py`, and my wrapper for gaiamock in `./utils/GaiamockWrapperLite.py`. 

The first step for setting this up is having `Gaiamock` somewhere on your machine, and putting its directory in `config.json` so the scripts can find it.

`GenerateSyntheticDatasets.ipynb` is a notebook that shows how I interact with my script and make my synthetic datasets, changing the distributions and different parameters. It's not trivial but also not a nightmare to swap in different choices for period, mass ratios, and eccentricities. `SyntheticData.py` should hopefully be commented enough. The parallelisation is a bit terse, but just ignore that.
