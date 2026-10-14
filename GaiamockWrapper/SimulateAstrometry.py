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

import GaiamockWrapper.Wrapper as gw
from GaiamockWrapper.utils import *

# =============================
# joblib + tqdm integration
# =============================
@contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar.
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
    if using_gaiamock:
        return func(*params, c_funcs=c_funcs, **kwargs)
    return func(*params, **kwargs)

# =============================
# Main generator
# =============================

def simulate_population_astrometry(object_count, catalogue, binary_fraction, using_gaiamock=True,
                                   
                                    period_model=normal_distribution, period_kwargs={"gaussian_params":(5, 3), "lims":(1, 8), "log":True}, # log-normal Moe+DiStefano 2017
                                    companion_mass_model=uniform_distribution, m_kwargs={"lims": [0.1, 1]}, # flat mass ratio distribution 
                                    compute_mass_ratios=True, # whether to compute mass ratios, or directly compute companion masses
                                    eccentricity_model=power_law_distribution, ecc_kwargs={"index": 1, "lims": [0, 0.99]}, # thermal eccentricity distribution
                                    flux_ratio_model=flux_ratio_power_law, f_kwargs={"index": 4}, # mass-luminosity relationship
                                    
                                    reassign_parallaxes=False, # whether to sample sources and then adjust parallax to some chosen range
                                    parallax_model=generate_parallax, plx_kwargs={"dist_range": (100, 200)}, 
                                    
                                    astrometry_function=gw.astrometry_wrapper,
                                    
                                    data_release="dr3", verbose=True, n_jobs=-1,
                                    astrometry_columns=_ASTROMETRY_COLUMNS, **kwargs):
    """
    Simulate a population of astrometric sources, including both single and binary systems.
    Parameters:
    - object_count: int, total number of objects to simulate
    - catalogue: astropy Table, input catalogue with required columns
    - binary_fraction: float, fraction of objects that are binaries
    - period_model: function, model to generate orbital periods for binaries
    - period_kwargs: dict, additional arguments for the period model
    - companion_mass_model: function, model to generate mass ratios for binaries
    - m_kwargs: dict, additional arguments for the mass ratio model
    - eccentricity_model: function, model to generate eccentricities for binaries
    - ecc_kwargs: dict, additional arguments for the eccentricity model
    - flux_ratio_model: function, model to generate flux ratios for binaries
    - f_kwargs: dict, additional arguments for the flux ratio model
    - astrometry_function: function, astrometry function for binaries
    - data_release: str, Gaia data release to use ("dr3" or "dr4")
    - mod: bool, whether to use the modified scanning law (True) or the standard scanning law (False)
    - verbose: bool, whether to print progress messages
    - n_jobs: int, number of parallel jobs to run (-1 for all available cores)
    - astrometry_columns: ordered numeric columns passed to the astrometry
      callbacks. By default this is gw.ASTROMETRY_FIELDS. Custom callbacks can
      request additional numeric columns by supplying their names here.
    - **kwargs: additional keyword arguments to pass to BOTH astrometry functions
    """

    # number of binaries to make
    binary_count = int(object_count*binary_fraction)
    
    # catalogue checks
    # relevant parameters: ra, dec, pmra, pmdec, mass, plx, gmag
    # check if they all exist
    if (not all(param in catalogue.colnames for param in ["ra", "dec", "pmra", "pmdec", "mass", "parallax", "phot_g_mean_mag"])):
        raise ValueError("Catalogue must contain the following columns: ra, dec, pmra, pmdec, mass, parallax, phot_g_mean_mag")
    
    if verbose:
        print("(1) Assigning companions to {} sources...".format(binary_count))
        
    # randomly select objects from the catalogue
    idx = np.random.choice(len(catalogue), object_count, replace=True)
    # and the subset that are binaries
    binary_idxs = idx[:binary_count]
    
    # create empty arrays to hold the parameters
    periods = np.full(object_count, np.nan)
    mass_ratios = np.full(object_count, np.nan)
    eccentricities = np.full(object_count, np.nan)
    flux_ratios = np.full(object_count, np.nan)
    inc = np.full(object_count, np.nan)
    w = np.full(object_count, np.nan)
    omega = np.full(object_count, np.nan)
    Tp = np.full(object_count, np.nan)

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
    Tp[:binary_count] = random_Tp(binary_count)
    
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
    reconstructed_catalogue[:binary_count]["phot_g_mean_mag"] -= 2.5 * np.log10(flux_ratios[:binary_count])
    
    # If the user supplied a target parallax range, adjust the parallaxes of the sources to match that range
    if reassign_parallaxes:
        old_plxs = reconstructed_catalogue["parallax"]
        reconstructed_catalogue["parallax"] = parallax_model(n=object_count, table=reconstructed_catalogue, **plx_kwargs)
        # adjust the parallaxes of the sources to match the new range
        reconstructed_catalogue["phot_g_mean_mag"] = adjust_magnitude(reconstructed_catalogue["phot_g_mean_mag"], old_plxs, reconstructed_catalogue["parallax"])

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
                                     data_release=data_release, is_binary=reconstructed_catalogue["is_binary"][i], **kwargs)
            for i in range(object_count))

    return reconstructed_catalogue, results

def run_astrometric_cascade(astrometry_set, cascade_func, using_gaiamock=True, verbose=True, **kwargs):
    """
    Run the astrometric cascade on a set of astrometry results and return the final solution types.
    """
    # compute the binaries
    if verbose:
        pbar = tqdm(total=len(astrometry_set), desc="Computing astrometry for all sources...")
            
    # run the astrometry function in parallel for all the sources, with a progress bar
    with tqdm_joblib(pbar if verbose else tqdm(disable=True)):
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_run_astrometry)(cascade_func, using_gaiamock, *astrometry_set[i], **kwargs)
            for i in range(len(astrometry_set)))