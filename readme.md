# Generating mock ${\it Gaia}$ datasets

Ben Pennell, MPIA

October 13th, 2026
_____

In ${\it Gaia}$, we will want to simulate mock populations of binaries just to see how the data would look, to explore if projects will be doable, or to properly forward-model and directly learn from the simulations. The key, fundamental tool is therefore the ability to create such mock data, and to have a system to do it in a modular way, so one could go from an envisioned population to a cursory look at the astrometric summary statistics, fits to certain models, rates, etc.

## Simulating binaries
### Choosing primaries
The first part is creating a set of primary stars, which I want to have possible through the following approaches:
1. Sampling a set of "primaries" either simulated or taken directly from ${\it Gaia}$ and randomly assigning companions and then computing the astrometry
    - And also taking a set of observed "primaries" and extrapolating to different distances by scaling the magnitudes and such
2. Some simple population synthesis, i.e sampling from an IMF 

### Choosing secondaries
The second part of simulating synthetic data is assigning companions. Basically, there are three relevant parameters: mass of the companion $M_2$, orbital period $P$, and eccentricity $e$. There's then the question of assigning a flux to the companion as well. What I propose is a system wherein each of the parameters can have functions supplied which can change their distributions. I'll create some "default" options, but then the user can supply different ones. The flow I'm thinking is that first an orbital period is assigned, with a function $f(P\mid M_1)$, then the companion mass can be assigned $f(M_2 \mid P, M_1)$, then the eccentricity $f(e\mid M_2, P, M_1)$ and finally a flux ratio $f(f\mid M_1, M_2)$.
After, orbital angles are assigned randomly.

The idea here is that I want to maintain the ability to have inseperable distributions, and doing so in this cascade I believe will have the simplest logical flow.

### ${\tt Gaiamock}$
Then, the sources are propagated through ${\tt Gaiamock}$. One thing of interest is the fact that ${\tt Gaiamock\_mod}$ exists, and I will allow the user to change between them. I would also like to allow the user to provide their own noise model based on the magnitude and colour of the source $f(\sigma\mid g, f)$.

## Astrometric models
With a set of mock ${\it Gaia}$ astrometric data, one would then want to fit to some set of models. The general idea is that the user will supply a ${\it cascade}$, or use the default DR4 option (which I'll add as soon as we know it). Basically, a ${\it cascade}$ has functions which compute some fits, assigns some parameters, and computes summary statistics for each. Then, there should be a second function which uses the summary statistic to determine which astrometric solution to keep. in DR4, this will be the 5,7,9,12 parameter models all computed at once and some cuts used to determine which to publish. In our personal endeavours, we may only construct a "is binary" and "is not binary" model, or something like this. Either way, the strategy is the same. What I would like to do is provide a way for the user to swap-in their own cascade. Additionally, I'd like to allow the user to swap-in their own treatment of the full orbit fits, since a particular choice is made in ${\tt Gaiamock}$ to do it a certain way.

Additionally, I will add an option to return the best-fit parameters from the selected model, and the summary statistics. This would allow, for example, the user to determine after creating the simulated catalogue where the cuts should be, and it can be seen exactly which sources "drop out" as cuts change.

### Features list:
General features:
- IMF/isochrone/mass-luminosity sampling (?)
- ${\it Gaia}$ preprocessing function (for DR3 for now)
- make it into a package (how to??)
- make a resampling function
- make documentation for all of this

Creating mock population:
- ~~make it just work in general~~
- ~~modular parameterisations~~
- ~~Propagate extra table columns through the machinery~~
- ~~Think about computing masses vs mass ratios?~~
- ~~Adjusting distance distribution~~
- A preprocessing option? Choice to pass in a function to eliminate sources that fail some kind of cut before the "astrometric cascade"
- ~~Make an adjustment for the magnitude based on the flux ratio~~
- Make an estimated col change from the companion?

Astrometric cascade:
- ~~modular cascade~~
- ~~return the summary statistics in some more mindful way~~
- ~~make a seperate function for just computing the solution types~~
- ~~make a stand-in for DR4~~
- an option to swap-in Sahar's linear one??
