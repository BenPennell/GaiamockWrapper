# The Wrapper

Ben Pennell, MPIA

October 13th, 2026

Version 0.2.0
_____

In ${\it Gaia}$, we will want to simulate mock populations of binaries just to see how the data would look, to explore if projects will be doable, or to properly forward-model things directly. The key and fundamental tool is therefore the ability to create such mock data, and to have a system to do it in a modular way, so one could rapidly (in both silicon and carbon time) go from an envisioned population to a cursory look at the astrometric summary statistics, fits to certain models, rates, etc.

# Installation
You need to install ${\tt Gaiamock}$ (https://github.com/kareemelbadry/gaiamock) and the modified version. The information is all available in the github. This is a package you should download locally, in addition to this wrapper.

You need to copy `config.json` into your working folder, where you should enter the paths to gaiamock and to this wrapper, as it shows in the example. You can see in the example scripts how to load in gaiamock and the wrapper, but basically you need to tell python to switch your path to be in the wrapper directory, then inside the wrapper it will change its directory to that of gaiamock

```
import json
import sys

with open('./config.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

sys.path.append(d["wrapper_path"])

import SimulateAstrometry, style, utils
```

Finally, if you want to do the primary popsynth business, you'll need to install the `isochrones` package. Check `2)LightPopsynthExample.ipynb` for instructions and an example.


# Simulating binaries
### Choosing primaries
The first part is creating a set of primary stars, which I have made possible through the following approaches:
1. Sampling a set of "primaries" either simulated or taken directly from ${\it Gaia}$ and randomly assigning companions and then computing the astrometry
2. Some simple population synthesis, i.e sampling from an IMF 
3. Taking a set of primaries (from one of the above schemes) and adjusting their spatial distributions (i.e. changing how far away they are)

### Choosing secondaries
The second part of simulating synthetic data is assigning companions. Basically, there are three relevant parameters: mass of the companion $M_2$, orbital period $P$, and eccentricity $e$. There's then the question of assigning a flux to the companion as well. What I propose is a system wherein each of the parameters can have functions supplied which can change their distribution.

I'll create some "default" options, but then the user can supply different ones. The flow I'm thinking is that first an orbital period is assigned, with a function $f(P\mid M_1)$, then the companion mass can be assigned $f(M_2 \mid P, M_1)$, then the eccentricity $f(e\mid M_2, P, M_1)$ and finally a flux ratio $f(f\mid M_1, M_2)$.
After, orbital angles are assigned randomly.
These functions are called hierarchically such that non-seperable distributions can be proposed.

### ${\tt Gaiamock}$, or whatever
Then, the sources are propagated through ${\tt Gaiamock}$. One thing of interest is the fact that ${\tt Gaiamock\_mod}$ exists, and I will allow the user to select which they use. I would also like to allow the user to provide their own noise model based on the magnitude and colour of the source $f(\sigma\mid g, f)$.

## Astrometric models
With a set of mock ${\it Gaia}$ astrometric data, one would then want to fit to some set of models. The general idea is that the user will supply a ${\it cascade}$, or use the default DR4 option (which I'll add as soon as we know it). Basically, a ${\it cascade}$ has functions which compute some fits, assigns some parameters, and computes summary statistics for each and some decision for the solution type. In our personal endeavours, we may only construct a "is binary" and "is not binary" model, or something like this. Either way, the strategy is the same. What I would like to do is provide a way for the user to swap-in their own cascade. Additionally, I'd like to allow the user to swap-in their own treatment of the full orbit fits, since a particular choice is made in ${\tt Gaiamock}$ to do it a certain way (simulated annealing).

Additionally, the user can request the summary statistics for all sources, such that after simulating the astrometry and making the model fits, the user can adjust how their cascade would select solution types, or troubleshoot and investigate the selection function.

# What's available
`./Wrapper/` contains the scripts for the wrapper.
1. `AstroWrapper.py` is the file which actually interacts with `gaiamock` and contains the astrometric cascades
2. `SimulateAstrometry.py` is the file which contains the true machinery: samping primaries and creating a set of companions and then propagating them through an astrometric cascade from `AstroWrapper.py`. The main operating function here is `simulate_population_astrometry()`, to be explained below
3. `utils.py` has all the simple functions: probability distributions, sampling, postprocessing, etc. 
4. `style.py` holds my preferred matplotlib options

## simulate_population_astrometry()
The best way to internalise this complicated function is to just look at it.
```
def simulate_population_astrometry(object_count, catalogue, binary_fraction=1.0, bf_kwargs=None,                     
        period_model=normal_distribution, period_kwargs={"gaussian_params":(5, 3), "lims":(1, 8), "log":True}, # log-normal Moe+DiStefano 2017
        companion_mass_model=uniform_distribution, m_kwargs={"lims": [0.1, 1]}, # flat mass ratio distribution 
        compute_mass_ratios=True, # whether to compute mass ratios, or directly compute companion masses
        eccentricity_model=power_law_distribution, ecc_kwargs={"index": 1, "lims": [0, 0.99]}, # thermal eccentricity distribution
        flux_ratio_model=flux_ratio_power_law, f_kwargs={"index": 4}, # mass-luminosity relationship
        adjust_fluxes=True, # whether to adjust the fluxes of the primaries based on the flux ratio
        reassign_parallaxes=False, # whether to sample sources and then adjust parallax to some chosen range
        parallax_model=generate_parallax, plx_kwargs={"dist_range": (100, 200)}, 
        
        astrometry_function=gw.astrometry_wrapper,
        
        using_gaiamock=True, verbose=True, n_jobs=-1,
        astrometry_columns=_ASTROMETRY_COLUMNS, **kwargs):
```
It optionally takes in functions for the period, mass, eccentricity, flux ratio and parallax distributions. The defaults are standard sets but you should put in your own. In `utils.py`, I created several "default" probability distributions which work with my machinery, including these default ones. If you wish to create your own follow the following template:
```
def new_distribution(n=1, parameter1=default, parameter2=default2, **kwargs):
```
Where `n=1` will be the number of values to generate. All parameters must be keywords. Each function should generate a pdf, and the way my functions work is to do cdf sampling using my `choose_value()` cdf sampling function.
One thing that will be passed into each function is the table holding all of the primaries. so, you could use the metallicities to set the masses, or the multiplicity fraction, or whatever you want to.

I have created three notebooks with examples of using this machinery and how to use `simulate_population_astrometry()`. 

1. `1)SimpleExamples.ipynb` shows a series of examples with the real set of solar-type primaries in DR3
    - The simplest model
    - Changing companion parameter distributions
    - Changing the spatial distribution
    - Sampling with companion mass instead of mass ratio
    - Supplying a non-constant multiplicity
    - Using the current DR4 (BH3-cuts) estimated scheme
2. `2)LightPopsynthExample.ipynb`
    - Using the `isochrones` package to simulate primaries
    - Populate those primaries realistically on the sky
    - Create the right data structure for my machinery
    - Propagate through the machinery for another example set of binary models 
    - Demonstrate how to use resampling to quickly check stuff
3. `3)BHCompanionExample.ipynb`
    - Using analytic models to create primaries
    - Converting computed luminosities to G-band magnitudes
    - Populate those primaries realistically on the sky
    - Create the right data structure for my machinery
    - Propagate through the machinery for another example set of binary models


## Do-to list:
General features:
- ~~IMF/isochrone/mass-luminosity sampling (?)~~ Oct 20
- ~~${\it Gaia}$ preprocessing function (for DR3 for now)~~ Oct 19
- make it into a package (how to??)
- ~~make a resampling function~~ Oct 20
- ~~make a scheme to generate ra/dec/pmra/pmdec~~ Oct 21
- Make system for loading in gaiamock through a function, same with the optional matplotlib styling
    - this is how it should actually work for the package

Creating mock population:
- ~~make it just work in general~~ Oct 16
- ~~modular parameterisations~~ Oct 16
- ~~Propagate extra table columns through the machinery~~ Oct 16
- ~~Think about computing masses vs mass ratios?~~ Oct 16
- ~~Adjusting distance distribution~~ Oct 16
- A preprocessing option? Choice to pass in a function to eliminate sources that fail some kind of cut before the "astrometric cascade"
- ~~Make an adjustment for the magnitude based on the flux ratio~~ Oct 19
- Make an estimated col change from the companion?
- ~~postprocessing to put the stuff into nice astropy tables~~ Oct 19
- ~~multiplicity fraction can be assigned by a function~~ Oct 19
- make it take in a set of already made binaries or something like that? I suppose you could just make that yourself

Astrometric cascade:
- ~~modular cascade~~ Oct 16
- ~~return the summary statistics in some more mindful way~~ Oct 16
- ~~make a seperate function for just computing the solution types~~ Oct 16
- ~~make a stand-in for DR4~~ Oct 19
- an option to swap-in Sahar's linearised one??