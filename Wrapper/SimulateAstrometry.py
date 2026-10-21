# these guys are damn annoying on my machine so I shut them up
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

warnings.filterwarnings("ignore", message=".*unable to import Axes3D.*", category=UserWarning)

### IMPORTS
# for progress bars
try:
    # for Jupyter
    from tqdm.notebook import tqdm
except ImportError:
    # for terminal
    from tqdm import tqdm
    
from joblib import Parallel, delayed, parallel
from contextlib import contextmanager

import Wrapper.AstroWrapper as gw
from Wrapper.utils import *

# =============================
# joblib + tqdm integration
# =============================
@contextmanager
def tqdm_joblib(tqdm_object):
    """
    Connect joblib completion callbacks to a tqdm progress bar.

    Parameters:
        tqdm_object: (tqdm) tqdm progress bar to advance as jobs finish.

    Yields:
        tqdm_object: (tqdm) the same progress bar while the parallel block runs.
    """
    class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()

# =============================
# Cache the scanning law for all objects so that it doesn't need to be constantly recalculated
# Round the RA/Dec to a certain number of decimal places to reduce the number of unique positions
# Significantly speeds things up (~8x?)
# =============================

c_funcs = gw.generate_cfuncs()

_ASTROMETRY_COLUMNS = gw.ASTROMETRY_FIELDS

def _run_astrometry(func, using_gaiamock, *params, **kwargs):
    """
    Run an astrometry function. 

    Parameters:
        func: (callable) astrometry function to call.
        using_gaiamock: (bool) if True, we're using gaiamock so we need the c functions
        *params: (tuple) positional arguments for the function.
        **kwargs: (dict) keyword arguments for the function.

    Returns:
        result: (Any) output of func.
    """
    if using_gaiamock:
        return func(*params, c_funcs=c_funcs, **kwargs)
    return func(*params, **kwargs)

# =============================
# Main generator
# =============================

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
    """
    Simulate a population of stars and binaries and compute their astrometry.

    Parameters:
        object_count: (int) total number of sources to generate.
        catalogue: (Table) input source catalogue with required columns.
        binary_fraction: (float or callable) fraction of sources to assign binary parameters, or a function which returns a probability for a given source.
        bf_kwargs: (dict) extra arguments for the binary fraction function.
        using_gaiamock: (bool) if True, use the cached Gaia mock callback functions.
        period_model: (callable) function to draw binary orbital periods.
        period_kwargs: (dict) extra arguments for the period model.
        companion_mass_model: (callable) function to draw mass ratios or companion masses.
        m_kwargs: (dict) extra arguments for the companion mass model.
        compute_mass_ratios: (bool) if True, treat companion_mass_model as returning q values.
        eccentricity_model: (callable) function to draw eccentricities.
        ecc_kwargs: (dict) extra arguments for the eccentricity model.
        flux_ratio_model: (callable) function to draw flux ratios.
        f_kwargs: (dict) extra arguments for the flux ratio model.
        adjust_fluxes: (bool) if True, adjust the fluxes of the primaries based on the flux ratio.
        reassign_parallaxes: (bool) if True, replace parallax values with a new sampled range.
        parallax_model: (callable) function used to generate replacement parallaxes.
        plx_kwargs: (dict) extra arguments for the parallax model.
        astrometry_function: (callable) astrometry routine to evaluate each source.
        verbose: (bool) if True, print status and show a progress bar.
        n_jobs: (int) number of parallel workers to use.
        astrometry_columns: (list/tuple) column names passed as the astrometry input array.
        **kwargs: (dict) extra arguments passed to the astrometry function.

    Returns:
        reconstructed_catalogue: (Table) sampled catalogue with binary properties added.
        results: (list) astrometry outputs for each source.
    """
    
    # relevant parameters: ra, dec, pmra, pmdec, mass, plx, gmag
    # check if they all exist
    if (not all(param in catalogue.colnames for param in ["ra", "dec", "pmra", "pmdec", "mass", "parallax", "phot_g_mean_mag"])):
        raise ValueError("Catalogue must contain the following columns: ra, dec, pmra, pmdec, mass, parallax, phot_g_mean_mag")
    
    # randomly select objects from the catalogue
    idx = np.random.choice(len(catalogue), object_count, replace=True)
    # create empty arrays to hold the parameters
    periods = np.full(object_count, np.nan)
    mass_ratios = np.full(object_count, np.nan)
    eccentricities = np.full(object_count, np.nan)
    flux_ratios = np.full(object_count, np.nan)
    inc = np.full(object_count, np.nan)
    w = np.full(object_count, np.nan)
    omega = np.full(object_count, np.nan)
    Tp = np.full(object_count, np.nan)
    
    ###############
    if isinstance(binary_fraction, (float, int)):
        # is a binary fraction (number) is supplied, simply compute that number of binaries
        if not (0 <= binary_fraction <= 1):
            raise ValueError("binary_fraction must be between 0 and 1")
        # number of binaries to make
        binary_count = int(object_count*binary_fraction)
            
        if verbose:
            print("(1) Assigning companions to {} sources...".format(binary_count))
            
        # and the subset that are binaries
        binary_idxs = idx[:binary_count]
    ###############
    elif callable(binary_fraction):
        # if a function is supplied, compute the probability for each source and then draw a random number to determine if it's a binary
        if verbose:
            print("(1) Assigning companions to sources based on binary fraction function...")
        # determine probability of being a binary for each source
        probabilities = binary_fraction(catalogue[idx], **(bf_kwargs if bf_kwargs is not None else {}))
        # determine which are binaries based on the probabilities
        is_binary = np.random.rand(object_count) < probabilities
        # sort all the binaries to the front of the catalogue to make consistent with the set-rate case above
        binary_count = np.sum(is_binary)
        if verbose:
            print("(1) Assigned {} binaries, for an overall binary fraction of {:2f}\%".format(binary_count, binary_count/object_count*100))
        idx = np.concatenate([idx[is_binary], idx[~is_binary]])
        binary_idxs = idx[:binary_count]
    else:
        raise ValueError("binary_fraction must be a float or callable function")
    
    # generate the orbital parameters hierarchically for the binaries
    # --- periods ---
    periods[:binary_count] = period_model(n=binary_count, table=catalogue[binary_idxs], **period_kwargs)
    # --- mass ratios ---
    if compute_mass_ratios:
        mass_ratios[:binary_count] = companion_mass_model(n=binary_count, table=catalogue[binary_idxs], periods=periods[:binary_count], **m_kwargs) # directly compute mass ratios
    else:
        mass_ratios[:binary_count] = companion_mass_model(n=binary_count, table=catalogue[binary_idxs], periods=periods[:binary_count], **m_kwargs) / catalogue[binary_idxs]["mass"] # first compute companion masses
    # --- eccentricities ---
    eccentricities[:binary_count] = eccentricity_model(n=binary_count, table=catalogue[binary_idxs], periods=periods[:binary_count], mass_ratios=mass_ratios[:binary_count], **ecc_kwargs)
    # --- flux ratios ---
    flux_ratios[:binary_count] = flux_ratio_model(n=binary_count, table=catalogue[binary_idxs], mass_ratios=mass_ratios[:binary_count], **f_kwargs)
    
    # set the orbital angles
    # --- orbital angles ---
    inc[:binary_count] = random_inc(binary_count)
    w[:binary_count] = random_angle(binary_count)
    omega[:binary_count] = random_angle(binary_count)
    Tp[:binary_count] = random_Tp(binary_count)*periods[:binary_count]
    
    # --- reconstruct catalogue ---
    reconstructed_catalogue = catalogue[idx]
    reconstructed_catalogue["period"] = periods
    reconstructed_catalogue["m2"] = mass_ratios * reconstructed_catalogue["mass"]
    reconstructed_catalogue["q"] = mass_ratios
    reconstructed_catalogue["ecc"] = eccentricities
    reconstructed_catalogue["f"] = flux_ratios
    reconstructed_catalogue["inc"] = inc
    reconstructed_catalogue["w"] = w
    reconstructed_catalogue["omega"] = omega
    reconstructed_catalogue["Tp"] = Tp
    reconstructed_catalogue["is_binary"] = np.zeros(object_count, dtype=bool)
    reconstructed_catalogue["is_binary"][:binary_count] = True
    
    # rescale our magnitude based on the flux ratio
    # just by a simply M-L relationship. We won't think too hard
    # M = -2.5 log10(F) => dM = -2.5 log10(1 + f)
    if adjust_fluxes:
        reconstructed_catalogue[:binary_count]["phot_g_mean_mag"] -= 2.5 * np.log10(1 + flux_ratios[:binary_count])
    
    # If the user supplied a target parallax range, adjust the parallaxes of the sources to match that range
    if reassign_parallaxes:
        old_plxs = reconstructed_catalogue["parallax"]
        reconstructed_catalogue["parallax"] = parallax_model(n=object_count, table=reconstructed_catalogue, **plx_kwargs)
        # adjust the parallaxes of the sources to match the new range
        reconstructed_catalogue["phot_g_mean_mag"] = adjust_magnitude(reconstructed_catalogue["phot_g_mean_mag"], old_plxs, reconstructed_catalogue["parallax"])
        
        # re-adjust proper motions (linearly is good enough)
        reconstructed_catalogue["pmra"] = reconstructed_catalogue["pmra"] * (reconstructed_catalogue["parallax"] / old_plxs)
        reconstructed_catalogue["pmdec"] = reconstructed_catalogue["pmdec"] * (reconstructed_catalogue["parallax"] / old_plxs)
        
        bright_enough_mask = reconstructed_catalogue["phot_g_mean_mag"] < 19
        if verbose:
            print("(1) Reassigned parallaxes and adjusted magnitudes. {} sources G>19 were removed.".format(np.sum(~bright_enough_mask)))
        reconstructed_catalogue = reconstructed_catalogue[bright_enough_mask] # remove any sources that are now too faint
        object_count = len(reconstructed_catalogue)

    # =============================
    # Now actually go and compute the astrometry
    # =============================
    if verbose:
        print("(2) Computing astrometry for {} sources...".format(object_count))
    # Create minimal catalogues with only the relevant parameters for the astrometry
    astrometry_catalogue = np.column_stack([np.asarray(reconstructed_catalogue[column]) for column in astrometry_columns]).astype(float, copy=False)
    
    # compute the binaries
    if verbose:
        pbar = tqdm(total=object_count, desc="Computing astrometry for all sources...")
          
    # run the astrometry function in parallel for all the sources, with a progress bar
    with tqdm_joblib(pbar if verbose else tqdm(disable=True)):
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_run_astrometry)(astrometry_function, using_gaiamock, astrometry_catalogue[i], 
                                    is_binary=reconstructed_catalogue["is_binary"][i], **kwargs)
            for i in range(object_count))

    return reconstructed_catalogue, results

def run_astrometric_cascade(astrometry_set, cascade_func, using_gaiamock=True, verbose=True, n_jobs=-1, **kwargs):
    """
    Run a per-source astrometry cascade across a collection of inputs.

    Parameters:
        astrometry_set: (sequence) list of argument tuples to pass to cascade_func.
        cascade_func: (callable) function to evaluate for each source.
        using_gaiamock: (bool) if True, inject the cached Gaia mock callback functions.
        verbose: (bool) if True, show a progress bar.
        n_jobs: (int) number of parallel workers to use.
        **kwargs: (dict) extra arguments passed to cascade_func.

    Returns:
        results: (list) output from cascade_func for each item in astrometry_set.
    """
    if verbose:
        pbar = tqdm(total=len(astrometry_set), desc="Computing astrometry for all sources...")

    with tqdm_joblib(pbar if verbose else tqdm(disable=True)):
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_run_astrometry)(cascade_func, using_gaiamock, *astrometry_set[i], **kwargs)
            for i in range(len(astrometry_set)))

    return results