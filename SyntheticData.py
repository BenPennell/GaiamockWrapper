# =============================
# Silence noisy deprecation warnings
# =============================
import warnings
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API"
)

warnings.filterwarnings(
    "ignore",
    message=".*unable to import Axes3D.*",
    category=UserWarning,
)

# =============================
# Standard imports
# =============================
import numpy as np
from functools import lru_cache
from joblib import Parallel, delayed, parallel
from contextlib import contextmanager
from tqdm.auto import tqdm
import json
import sys
import os
from utils.utils import *

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
# Configuration / Imports
# =============================

with open("config.json") as f:
    d = json.load(f)

folder_a_path = os.path.abspath(os.path.join(os.getcwd(), d["wrapperlitepath"]))
if folder_a_path not in sys.path:
    sys.path.append(folder_a_path)

import GaiamockWrapperLite as gw

# =============================
# Functions for randomly setting orbital angles
# =============================

def random_angle(n=1):
    x = np.random.rand(n) * 2 * np.pi
    return float(x[0]) if n == 1 else x

def random_inc(n=1):
    x = np.arccos(2 * np.random.rand(n) - 1)
    return float(x[0]) if n == 1 else x

def random_Tp(n=1):
    x = np.random.rand(n) - 0.5
    return float(x[0]) if n == 1 else x

# =============================
# period, mass ratio, and eccentricity distributions
# =============================

# eccentricities
es = np.linspace(0, 1, 100)

e_pdf = np.zeros_like(es)
e_pdf[1:] = pexp(es[1:], 1)
e_cdf = np.cumsum(e_pdf / np.sum(e_pdf))

p_therm = pexp(es, 1, val_range=(es[0], es[-1]))
p_therm /= np.sum(p_therm)

rayleigh_params = (0.38, 0.2)
p_gaus = gaussian(es, *rayleigh_params)
p_gaus /= np.sum(p_gaus)

periods_grid = np.linspace(1, 8, 100)
turnover_params = (3.5, 1)
turnover_pdf = gaussian(periods_grid, *turnover_params)
turnover_weight = np.cumsum(turnover_pdf / np.sum(turnover_pdf))

def circular_e(*args):
    return 0.0

def thermal_e(*args):
    return np.interp(np.random.rand(), e_cdf, es)

def turnover_e(logP):
    w = turnover_weight[np.argmin(np.abs(periods_grid - logP))]
    dist = (1 - w) * p_gaus + w * p_therm
    cdf = np.cumsum(dist / np.sum(dist))
    return np.interp(np.random.rand(), cdf, es)

def choose_value(cdf, grid, size):
    u = np.random.uniform(cdf.min(), cdf.max(), size)
    return np.interp(u, cdf, grid)

# =============================
# Cache the scanning law for all the object
# Round the RA/Dec to a certain number of decimal places to reduce the number of unique positions
# Significantly speeds things up
# =============================

c_funcs = gw.generate_cfuncs()

_GOST_NDP = 5  # RA/Dec rounding

@lru_cache(maxsize=None)
def _get_gost_cached(ra_r, dec_r):
    return gw.gaiamock.get_gost_one_position(
        ra_r, dec_r, data_release="dr3"
    )

def get_gost(ra, dec):
    return _get_gost_cached(
        round(float(ra), _GOST_NDP),
        round(float(dec), _GOST_NDP),
    )

# =============================
# Function for fetching the scanning law and getting the solution type for a binary
# =============================

def solve_binary(period, q, ecc, inc, w, omega, Tp,
                ra, dec, pmra, pmdec, plx, mass, gmag):
    t = get_gost(ra, dec)
    return gw.rapid_solution_type(
        period, q, plx, mass,
        gmag, 1e-10, ecc,
        inc, w, omega, Tp,
        ra, dec, pmra, pmdec,
        t, c_funcs
    )

# =============================
# Main generator
# =============================

def create_synthetic_data(object_count, catalogue,
                        binary_fraction=None, binarity_model=None, mass_model=None, period_model=None, ecc_type="circular",
                        m_lim=(0.013, 0.2), q_lim=(0.05, 0.5), p_lim=(1, 8), p_resolution=100, 
                        save_bprp=True, verbose=True, n_jobs=-1):
    
    '''
    Create a synthetic dataset of Gaia-like objects, with a specified fraction of binaries and various models for the binary parameters.
    Parameters:
        - object_count: total number of objects to generate
        - catalogue: Ideally an astropy table containing the properties of the objects to sample from (must include ra, dec, pmra, pmdec, parallax, mass_single, phot_g_mean_mag, and optionally bp_rp)
            Should be something that can be indexed like `catalogue["ra"]` to extract all right ascensions
        - binary_fraction: if not None, the fixed fraction of objects that are binaries (overrides binarity_model)
        - binarity_model: a function that takes an array of masses and returns an array of probabilities of being binary (if binary_fraction is None)  
            Use this if you want some kind of mass-dependent binary fraction or whatever.  
        - mass_model: a parameter for the mass ratio distribution (power law exponent). If None, you get a flat distribution
        - period_model: a tuple (mu, sigma) for a Gaussian distribution of log(period). if None, you get a flat distribution
        - ecc_type: one of "circular", "thermal", or "turnover" to specify the eccentricity distribution
        - m_lim: tuple (m_min, m_max) specifying the allowed range of secondary masses (in solar masses)
        - q_lim: tuple (q_min, q_max) specifying the allowed range of mass ratios (m2/m1)
        - p_lim: tuple (p_min, p_max) specifying the allowed range of log(period) (in days)
        - p_resolution: number of points to use in the period grid if period_model is specified. You don't need to change this.
        - save_bprp: whether to include the bp_rp color in the output (if available in the catalogue). Defaults to True.
        - verbose: whether to show a progress bar during the binary solving step
        - n_jobs: number of parallel jobs to use for solving

    Returns:
        A numpy array of length object_count, where each element is a dictionary containing the properties of the object, including the binary parameters if it is a binary.
        Each object gets the field "is_binary" which is True for binaries and False for singles, and "solution_type" which is 0,5,7,9, or 12.
    '''

    # choose which of the three eccentricity functions is called for
    ecc_func = {"circular": circular_e, "thermal": thermal_e, "turnover": turnover_e,}.get(ecc_type, circular_e)

    def flat_q(count):
        return np.random.uniform(*q_lim, count)
    q_func = flat_q
    # if a mass ratio distribution is called for (power law), set up the function
    if mass_model is not None:
        qs = np.linspace(*q_lim, 1000)
        q_pdf = pexp(qs, mass_model)
        q_cdf = np.cumsum(q_pdf / np.sum(q_pdf))
        def exponential_q(count):
            return np.array([np.interp(np.random.rand(), q_cdf, qs) for _ in range(count)])   
        q_func = exponential_q
        
    # --- select catalogue rows ---
    # weight by volume coverage
    # for the less complete ones, we want more objects corresponding to the relative enclosed volume
    # areas = np.array([relative_volume(catalogue[i]["phot_g_mean_mag"], catalogue[i]["parallax"]) for i in range(len(catalogue))])
    # idx = np.random.choice(len(catalogue), object_count, p=1/areas/np.sum(1/areas)) # normalised probability is the inverse of the covered area

    # randomly select objects from the catalogue
    idx = np.random.choice(len(catalogue), object_count, replace=True)
    ra = catalogue["ra"][idx].astype(float)
    dec = catalogue["dec"][idx].astype(float)
    pmra = catalogue["pmra"][idx].astype(float)
    pmdec = catalogue["pmdec"][idx].astype(float)
    mass = catalogue["mass_single"][idx].astype(float)
    plx = catalogue["parallax"][idx].astype(float)
    gmag = catalogue["phot_g_mean_mag"][idx].astype(float)
    if save_bprp:
        bprp = catalogue["bp_rp"][idx].astype(float)
    
    # choose some amount of objects to be binaries
    # --- binary mask ---
    if binary_fraction is not None:
        p_bin = np.full(object_count, binary_fraction, dtype=float)
    else: # use supplied model function
        p_bin = np.asarray(binarity_model(mass), dtype=float)
    binary_mask = np.random.rand(object_count) < p_bin    
        
    bin_idx = np.where(binary_mask)[0]
    nb = len(bin_idx)

    # --- periods ---
    if period_model is not None:
        mu, si = period_model
        ps = np.linspace(*p_lim, p_resolution)
        pdf = gaussian(ps, mu, si)
        cdf = np.cumsum(pdf / pdf.sum())
        logP = choose_value(cdf, ps, nb)
    else:
        logP = np.random.uniform(p_lim[0], p_lim[1], nb)

    period = 10 ** logP

    # --- mass ratios ---
    # randomly select mass ratios, and then keep resampling
    # until they fall into the restricted range 
    q = q_func(nb)
    m2 = q * mass[bin_idx]
    bad = (m2 < m_lim[0]) | (m2 > m_lim[1])
    while np.any(bad):
        q[bad] = q_func(bad.sum())
        m2[bad] = q[bad] * mass[bin_idx][bad]
        bad = (m2 < m_lim[0]) | (m2 > m_lim[1])

    # --- eccentricities ---
    ecc = np.array([ecc_func(lp) for lp in logP])
    
    # --- orbital angles ---
    inc = random_inc(nb)
    w = random_angle(nb)
    omega = random_angle(nb)
    Tp = random_Tp(nb) * period

    # =============================
    # Parallel solve with progress bar
    # =============================

    if verbose:
        pbar = tqdm(total=nb, desc="Computing Binaries")

    with tqdm_joblib(pbar if verbose else tqdm(disable=True)):
        results = Parallel(
            n_jobs=n_jobs,
            backend="loky"
        )(
            delayed(solve_binary)(
                period[b], q[b], ecc[b], inc[b], w[b], omega[b], Tp[b],
                ra[i], dec[i], pmra[i], pmdec[i],
                plx[i], mass[i], gmag[i]
            )
            for b, i in enumerate(bin_idx)
        )

    # =============================
    # Assemble output
    # =============================

    outdata = []
    b = 0
    for i in range(object_count):
        out = {
            "ra": ra[i],
            "dec": dec[i],
            "pmra": pmra[i],
            "pmdec": pmdec[i],
            "parallax": plx[i],
            "mass": mass[i],
            "phot_g_mean_mag": gmag[i],
            "is_binary": bool(binary_mask[i]),
            "solution_type": 0,
        }
        if save_bprp:
            out["bp_rp"] = bprp[i]

        if binary_mask[i]:
            out.update({
                "period": period[b],
                "m2": m2[b],
                "q": q[b],
                "ecc": ecc[b],
                "inc": inc[b],
                "w": w[b],
                "omega": omega[b],
                "Tp": Tp[b],
                "solution_type": results[b],
            })
            b += 1

        outdata.append(out)

    return np.array(outdata)
