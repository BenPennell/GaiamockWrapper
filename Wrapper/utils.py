import numpy as np
import pandas as pd
from astropy.table import Table
from IPython.display import display
from scipy.special import gamma

### DEFAULT RANDOM ORBITAL ANGLE GENERATORS ###
def random_angle(n=1):
    """
    Draw random orbital angles in the range [0, 2pi).

    Parameters:
        n: (int) number of angles to sample.

    Returns:
        x: (float or ndarray) sampled angle(s) in radians.
    """
    x = np.random.rand(n) * 2 * np.pi
    return float(x[0]) if n == 1 else x

def random_inc(n=1):
    """
    Draw random inclinations from a distribution flat in cos(i).

    Parameters:
        n: (int) number of inclinations to sample.

    Returns:
        x: (float or ndarray) sampled inclination(s) in radians.
    """
    x = np.arccos(2 * np.random.rand(n) - 1) # Flat in cos(i)
    return float(x[0]) if n == 1 else x

def random_Tp(n=1):
    """
    Draw random time-of-periapsis offsets between 0 and 1 (fraction of orbital period).

    Parameters:
        n: (int) number of offsets to sample.

    Returns:
        x: (float or ndarray) sampled offset(s).
    """
    x = np.random.rand(n)
    return float(x[0]) if n == 1 else x

def kms_to_mas(kms, plx):
    """
    Convert proper motion from km/s to mas/yr using parallax (approximately lol)

    Parameters:
        kms: (float or ndarray) proper motion in km/s.
        plx: (float or ndarray) parallax in mas.
        
    Returns:
        mas: (float or ndarray) proper motion in mas/yr.
    """
    return kms * 4.74 * plx

def random_sky_positions(table, dispersion=30):
    """
    Randomly assign sky positions (RA, Dec) and proper motions (pmra, pmdec) to a table somewhat realistically.
    Follow (2)LightPopsynthExample.ipynb to see the usage. the table needs to have parallax!!
    
    RA/DEC will be assigned randomly, while the proper motion will be assigned based on a given velocity dispersion in km/s.
    30km/s dispersion is reasonable. but idk do your own research
    
    Parameters:
        table: (Table) Astropy Table containing source data.
        dispersion: (float) Velocity dispersion in km/s.
        
    Returns:
        None (updates the table in place).
    """
    # randomly assign sky positions
    table["ra"] = np.random.uniform(0, 360, len(table))
    table["dec"] = np.random.uniform(-90, 90, len(table))
    
    # for each star, generate velocities from a normal distribution and convert to mas/yr using the parallax
    proper_motion = np.random.normal(0, dispersion, len(table)) # [km/s]
    table["pmra"] = kms_to_mas(proper_motion * np.cos(np.radians(table["dec"])), table["parallax"]) # [mas/yr]
    table["pmdec"] = kms_to_mas(proper_motion * np.sin(np.radians(table["dec"])), table["parallax"]) # [mas/yr]

### RANDOM HELPER FUNCTIONS ###
### --- ###
def gaussian(x, mu, sigma):
    """
        Normalised gaussian at x, defined by two
        parameters: peak (mu) and width (sigma)
    
    Parameters:
        x: (float or ndarray) value(s) at which to evaluate the Gaussian.
        mu: (float) mean of the Gaussian.
        sigma: (float) standard deviation of the Gaussian.

    Returns:
        y: (float or ndarray) Gaussian probability density at x.
    """
    return np.exp(-(mu - x)**2/(2*sigma**2)) / np.sqrt(2 * np.pi * sigma**2)

### --- ###
def area_in_range(target_range, mu, sigma, resolution=100):
    """
    Compute the area under a Gaussian between two values.

    Parameters:
        target_range: (tuple) lower and upper bounds of the integration range.
        mu: (float) mean of the Gaussian.
        sigma: (float) standard deviation of the Gaussian.
        resolution: (int) number of grid points used for the integral.

    Returns:
        area: (float) area under the Gaussian over the requested range.
    """
    xs = np.linspace(*target_range, resolution)
    ys = gaussian(xs, mu, sigma)
    return np.trapezoid(y=ys, x=xs)

### --- ###
def pexp(val, index, val_range=(0, 1), ignore_a=False):
    """
        normalised power law probability
    
    Parameters:
        val: (float or ndarray) value(s) at which to evaluate the power law.
        index: (float) power-law exponent.
        val_range: (tuple) range over which the distribution is normalised.
        ignore_a: (bool) if True, omit the normalisation constant.

    Returns:
        p: (float or ndarray) power-law probability density value(s).
    """
    a = 1
    if not ignore_a:
        a = (index + 1) / (val_range[1] ** (index + 1) - val_range[0] ** (index + 1))
    return a * (val ** index)

### -- ###
def area_in_range_powerlaw(target_range, index, resolution=100):
    """
    Compute the area under a power-law distribution over a range.

    Parameters:
        target_range: (tuple) lower and upper bounds of the integration range.
        index: (float) power-law exponent.
        resolution: (int) number of grid points used for the integral.

    Returns:
        area: (float) area under the power law over the requested range.
    """
    xs = np.linspace(*target_range, resolution)
    ys = pexp(xs, index, ignore_a=True)
    return np.trapezoid(y=ys, x=xs)

### --- ###
def cutoff_to_fraction(p_model, pcut, resolution=100):
    """
    Convert a period cutoff into the fraction of observable systems above that cutoff.

    Parameters:
        p_model: (tuple) Gaussian parameters (mu, sigma) for the period distribution.
        pcut: (float) cutoff period value.
        resolution: (int) number of grid points used for integration.

    Returns:
        fraction: (float) observable fraction at the given cutoff.
    """
    p_mu, p_si = p_model
    total_area = area_in_range((1,pcut), p_mu, p_si, resolution)
    observable_area = area_in_range((2,3), p_mu, p_si, resolution)
    return observable_area / total_area

### --- ###
def fraction_to_cutoff(p_model, fraction, resolution=100):
    """
    Find the period cutoff that yields a target observable fraction.

    Parameters:
        p_model: (tuple) Gaussian parameters (mu, sigma) for the period distribution.
        fraction: (float) target observable fraction.
        resolution: (int) number of grid points used for integration.

    Returns:
        pcut: (float) cutoff period corresponding to the requested fraction.
    """
    p_mu, p_si = p_model
    observable_area = area_in_range((2,3), p_mu, p_si, resolution=resolution)
    target_area = observable_area / fraction
    # search for cutoff
    pcut_vals = np.linspace(3,8,1000)
    for pcut in pcut_vals:
        total_area = area_in_range((1,pcut), p_mu, p_si, resolution=resolution)
        if total_area >= target_area:
            return pcut
    return 8.0

### DEFAULT ORBITAL PARAMETER DISTRIBUTIONS ###
# you can follow how these are created when making your own
### --- ###
def choose_value(cdf, grid, size):
    """
        Helper function to sample from a distribution defined by a cdf and grid of values
    
    Parameters:
        cdf: (array-like) cumulative distribution function values.
        grid: (array-like) values corresponding to the cdf grid.
        size: (int) number of samples to draw.

    Returns:
        samples: (ndarray) sampled values from the distribution.
    """
    u = np.random.uniform(cdf.min(), cdf.max(), size)
    return np.interp(u, cdf, grid)

def return_distribution(vals, log=False, logbase=10, **kwargs):
    """
        Helper function which processes the output of a distribution function to return either log or linear values
    
    Parameters:
        vals: (float or ndarray) sampled values from a distribution.
        log: (bool) if True, transform values to log space.
        logbase: (float) base of the logarithm used for the transformation.
        **kwargs: (dict) unused keyword arguments kept for interface compatibility.

    Returns:
        vals: (float or ndarray) values in either linear or log space.
    """
    if log:
        vals = logbase ** vals
    return vals

### simple distributions ###
def constant_distribution(n=1, value=2.5, **kwargs):
    """
    Return a constant value repeated n times.

    Parameters:
        n: (int) number of values to generate.
        value: (float) constant value to repeat.
        **kwargs: (dict) unused keyword arguments.

    Returns:
        vals: (ndarray) array of repeated constant values.
    """
    return np.full(n, value)

def uniform_distribution(n=1, lims=(2, 3), **kwargs):
    """
    Draw values from a uniform distribution between two limits.

    Parameters:
        n: (int) number of values to draw.
        lims: (tuple) lower and upper bounds of the uniform distribution.
        **kwargs: (dict) extra arguments passed to return_distribution.

    Returns:
        vals: (ndarray) uniformly sampled values.
    """
    ys = np.random.uniform(lims[0], lims[1], n)

    return return_distribution(ys, **kwargs)

def normal_distribution(n=1, gaussian_params=(5, 3), lims=(2, 3), resolution=1000, **kwargs):
    """
    Draw values from a Gaussian distribution truncated to a range.

    Parameters:
        n: (int) number of values to draw.
        gaussian_params: (tuple) Gaussian mean and standard deviation.
        lims: (tuple) lower and upper bounds of the truncated distribution.
        resolution: (int) number of grid points in the sampling grid.
        **kwargs: (dict) extra arguments passed to return_distribution.

    Returns:
        vals: (ndarray) sampled Gaussian values.
    """
    mu, si = gaussian_params
    xs = np.linspace(*lims, resolution)
    pdf = gaussian(xs, mu, si)
    cdf = np.cumsum(pdf / pdf.sum())
    ys = choose_value(cdf, xs, n)

    return return_distribution(ys, **kwargs)

def power_law_distribution(n=1, index=2, lims=(2, 3), resolution=100, **kwargs):
    """
    Draw values from a power-law distribution over a given range.

    Parameters:
        n: (int) number of values to draw.
        index: (float) power-law exponent.
        lims: (tuple) lower and upper bounds of the distribution.
        resolution: (int) number of grid points in the sampling grid.
        **kwargs: (dict) extra arguments passed to return_distribution.

    Returns:
        vals: (ndarray) sampled power-law values.
    """
    xs = np.linspace(*lims, resolution)
    pdf = pexp(xs, index, val_range=lims)
    cdf = np.cumsum(pdf / pdf.sum())
    ys = choose_value(cdf, xs, n)

    return return_distribution(ys, **kwargs)

def gamma_distribution(n=1, shape=2, scale=1, lims=(0, 10), resolution=100, **kwargs):
    """
    Draw values from a gamma distribution over a given range.
    Parameters:
        n: (int) number of values to draw.
        shape: (float) shape parameter of the gamma distribution.
        scale: (float) scale parameter of the gamma distribution.
        lims: (tuple) lower and upper bounds of the distribution.
        resolution: (int) number of grid points in the sampling grid.
        **kwargs: (dict) extra arguments passed to return_distribution.

    Returns:
        vals: (ndarray) sampled gamma values.
    """
    xs = np.linspace(*lims, resolution)
    pdf = xs** (shape - 1)* np.exp(-xs/ scale)/ (gamma(shape) * scale**shape)
    cdf = np.cumsum(pdf / pdf.sum())
    ys = choose_value(cdf, xs, n)

    return return_distribution(ys, **kwargs)

def flux_ratio_power_law(index=4, mass_ratios=None, **kwargs):
    """
    Compute the flux ratio from a mass-ratio power law.

    Parameters:
        index: (float) exponent of the flux-ratio relationship.
        mass_ratios: (float or ndarray) mass ratio(s) to raise to the power.
        **kwargs: (dict) unused keyword arguments.

    Returns:
        flux_ratios: (float or ndarray) flux ratio(s) implied by the mass ratio.
    """
    return mass_ratios ** index

### FUCNTIONS FOR RESCALING-SAMPLES TO DIFFERENT DISTANCES ###
### --- ###
def adjust_magnitude(mag, plx, new_plx):
    """Adjust the magnitude of a star to reflect a change in parallax.

    Parameters:
        mag: (float or ndarray) original apparent magnitude.
        plx: (float or ndarray) original parallax in mas.
        new_plx: (float or ndarray) new parallax in mas.

    Returns:
        new_mag: (float or ndarray) updated apparent magnitude.
    """
    # Calculate the absolute magnitude
    abs_mag = mag - 5 * np.log10(1000/plx) + 5
    
    # Calculate the new apparent magnitude
    new_mag = abs_mag + 5 * np.log10(1000/new_plx) - 5
    return new_mag

### --- ###
def adjust_parallax(plx, mag, new_mag):
    """Adjust the parallax of a star to reflect a change in magnitude.

    Parameters:
        plx: (float or ndarray) original parallax in mas.
        mag: (float or ndarray) original apparent magnitude.
        new_mag: (float or ndarray) target apparent magnitude.

    Returns:
        new_plx: (float or ndarray) updated parallax in mas.
    """
    # Calculate the absolute magnitude
    abs_mag = mag - 5 * np.log10(1000/plx) + 5
    
    # Calculate the new parallax
    new_plx = 10 * 10**((abs_mag - new_mag + 5) / 5)
    return new_plx

### --- ###
def generate_parallax(n=1, dist_range=(100,200), resolution=1000, **kwargs):
    """
    Draw parallaxes corresponding to distances sampled from a d^2 distribution.

    Parameters:
        n: (int) number of parallax values to generate.
        dist_range: (tuple) minimum and maximum distance values in parsecs.
        resolution: (int) number of grid points in the distance sampling grid.
        **kwargs: (dict) unused keyword arguments.

    Returns:
        parallax: (float or ndarray) sampled parallax values in mas.
    """
    # sample a distance from a distribution that goes as d^2 between 100 and 200
    # from inverse cdf sampling
    dists = np.linspace(*dist_range, resolution)
    d_pdf = np.zeros_like(dists)
    d_pdf[1:] = pexp(dists[1:], 2) # d^2
    d_cdf = np.cumsum(d_pdf / np.sum(d_pdf))
    d = np.interp(np.random.rand(n), d_cdf, dists)
    parallax = 1000 / d 
    return parallax

### PROCESSING FUNCTIONS ###
### --- ###
def postproces_solution_type(catalogue, astrometric_results):
    """
    Attach the solution type for each source to a catalogue.

    Parameters:
        catalogue: (Table) source catalogue to update.
        astrometric_results: (list) per-source astrometric results containing the solution type.

    Returns:
        None: updates the catalogue in place.
    """
    soltypes = np.zeros(len(catalogue), dtype=int)
    for i, result in enumerate(astrometric_results):
        soltypes[i] = result[0]
    catalogue["solution_type"] = soltypes

### --- ###
def postprocess_fits(astrometric_results):
    """
    Convert a list of astrometric result tuples into a structured Table.

    Parameters:
        astrometric_results: (list) per-source result tuples from the astrometry pipeline.

    Returns:
        outtable: (Table) structured output table of solution parameters and quality flags.
    """
    outtable = Table()
    cols=["solution_type", "ra_off", "dec_off", "parallax", "pmra", "pmdec",
            "period", "ecc", "Fp", "A", "B", "F", "G", "pmra_dot", "pmdec_dot", "pmra_ddot", "pmdec_ddot",
            "ruwe", "F2", "s", "parallax_over_error", "sig_ecc", "cut"]
    dtypes=[int, float, float, float, float, float, float, float, float, float, float, float, float,
                            float, float, float, float, float, float, float, float, float, "U20"]

    for i, col in enumerate(cols):
        outtable[col] = np.full(len(astrometric_results), np.nan, dtype=dtypes[i])
    
    outtable["solution_type"] = [result[0] for result in astrometric_results]
    for i, result in enumerate(astrometric_results):
        soltype = result[0]
        p0 = result[1]
        if p0 is not None:
            if soltype == 0 or soltype == 5:
                outtable["ra_off"][i] = p0[0]
                outtable["pmra"][i] = p0[1]
                outtable["dec_off"][i] = p0[2]
                outtable["pmdec"][i] = p0[3]
                outtable["parallax"][i] = p0[4]
            elif result[0] == 7:
                outtable["ra_off"][i] = p0[0]
                outtable["pmra"][i] = p0[1]
                outtable["pmra_dot"][i] = p0[2]
                outtable["dec_off"][i] = p0[3]
                outtable["pmdec"][i] = p0[4]
                outtable["pmdec_dot"][i] = p0[5]
                outtable["parallax"][i] = p0[6]
            elif result[0] == 9:
                outtable["ra_off"][i] = p0[0]
                outtable["pmra"][i] = p0[1]
                outtable["pmra_dot"][i] = p0[2]
                outtable["pmra_ddot"][i] = p0[3]
                outtable["dec_off"][i] = p0[4]
                outtable["pmdec"][i] = p0[5]
                outtable["pmdec_dot"][i] = p0[6]
                outtable["pmdec_ddot"][i] = p0[7]
                outtable["parallax"][i] = p0[8]
            elif result[0] == 12:
                outtable["ra_off"][i] = p0[0]
                outtable["dec_off"][i] = p0[1]
                outtable["parallax"][i] = p0[2]
                outtable["pmra"][i] = p0[3]
                outtable["pmdec"][i] = p0[4]
                outtable["period"][i] = p0[5]
                outtable["ecc"][i] = p0[6]
                outtable["Fp"][i] = p0[7]
                outtable["A"][i] = p0[8]
                outtable["B"][i] = p0[9]
                outtable["F"][i] = p0[10]
                outtable["G"][i] = p0[11]
        for stat in ["ruwe", "F2", "s", "parallax_over_error", "sig_ecc", "cut"]:
            if stat in result[3].keys():
                outtable[stat][i] = result[3][stat]
    return outtable

### --- ###
def print_table(table, label=None, dictionary=False, print_latex=False):
    """
    Pass in a table returned from postproces_solution_type() to print a clean table with the
    rates and number of each solution type.
    
    Behaves nicely with jupyter notebooks.
    
    Parameters:
        table: (Table) output table from postproces_solution_type().
        label: (str) optional caption for the table.
        dictionary: (bool) if True, return rates as a dictionary instead of a list.
        print_latex: (bool) if True, print the table in LaTeX format (so you could put it in overleaf directly).
    """
    
    if dictionary:
        rates = []
        for soltype in [0,5,7,9,12]:
            rates.append(len(table[[t["solution_type"] == soltype for t in table]])/len(table)*100)
    else:
        rates = [len(table[[t["solution_type"] == soltype for t in table]])/len(table)*100 for soltype in [0,5,7,9,12]]
    grid = [
        [100] + rates,
        [len(table)] + [int(grp/100*len(table)) for grp in rates]
    ]
    row_labels = ["Rate (%)", "Counts"]
    col_labels = ["All", "low RUWE", "high RUWE", "Acceleration", "Jerk", "Full Orbit"]
    df = pd.DataFrame(grid, index=row_labels, columns=col_labels)
    displaydf = df.style.format(
        "{:.0f}",
        subset=pd.IndexSlice["Counts", :]
    ).format(
        "{:.2f}",
        subset=pd.IndexSlice["Rate (%)", :]
    )
    
    if label is not None:
        displaydf.set_caption(label)
    
    if print_latex:
        rate_row = "Rate (\\%) & " + " & ".join(f"{r:.2f}" for r in rates) + r" \\"
        count_row = "Count & " + " & ".join(f"{int(r/100*len(table))}" for r in rates) + r" \\"
        print(rate_row)
        print(count_row)
    display(displaydf)
    
    
### DR3 PREPROCESSING FUNCTION ###
def DR3_preprocess_catalogue(table):
    """
        apply the DR3 preprocessing cuts to a sample taken directly from Gaia. Will update in DR4 (written 14/10/2026)
        
        n_visibility_periods > 12 Halbwachs+(2023)
        ipd_frac_multi_peak <= 2 Halbwachs+(2023)
        ipd_gof_harmonic_amplitude < 0.1 Halbwachs+(2023)
        f(bp_rp_excess, bp_rp) < g(phot_g_mean_mag) Riello+(2021)

    Parameters:
        table: (Table) Gaia DR3 catalogue to filter.

    Returns:
        keep: (array) boolean mask indicating which sources pass the preprocessing cuts.
    """
    keep = np.ones(len(table), dtype=bool) 
       
    req_cols = ["visibility_periods_used", "ipd_frac_multi_peak", "ipd_gof_harmonic_amplitude", "phot_bp_rp_excess_factor", "bp_rp", "phot_g_mean_mag"]
    
    #### VISIBILITY PERIODS
    if "visibility_periods_used" in table.colnames:
        keep[table["visibility_periods_used"] <= 12] = False
    else:
        print("WARNING: visibility_periods_used not in table, skipping this cut.")
    
    #### IPD FRAC MULTI PEAK
    if "ipd_frac_multi_peak" in table.colnames:
        keep[table["ipd_frac_multi_peak"] > 2] = False
    else:
        print("WARNING: ipd_frac_multi_peak not in table, skipping this cut.")

    #### IPD GOF HARMONIC AMPLITUDE
    if "ipd_gof_harmonic_amplitude" in table.colnames:
        keep[table["ipd_gof_harmonic_amplitude"] >= 0.1] = False
    else:   
        print("WARNING: ipd_gof_harmonic_amplitude not in table, skipping this cut.")
    
    #### BP/RP EXCESS FACTOR
    if "phot_bp_rp_excess_factor" in table.colnames and "bp_rp" in table.colnames and "phot_g_mean_mag" in table.colnames:
        ## CREATE THE Riello+(2021) CORRECTED BP-RP EXCESS FACTOR CUT
        bp_rps = table["bp_rp"]
        bres = table["phot_bp_rp_excess_factor"]
        apparent_mags = table["phot_g_mean_mag"]
        
        ## COMPUTE THE CORRECTED BP-RP EXCESS FACTOR
        ## THIS IS DONE WITH A SET OF POLYNOMIALS OF BP_RP (from a_mat)
        a_mat = np.array([[1.154360, 1.162004, 1.057572],
                    [0.033772, 0.011461, 0.140537],
                    [0.032277, 0.049255, 0.000000],
                    [0.000000, -0.005879, 0.000000]])
        bp_rp_edges = np.array([0.5, 4.0]) # define which column you're in
        bp_rp_cols = np.digitize(bp_rps, bp_rp_edges) # determine the oolumn for each source based on its BP-RP color
        a_vals = a_mat[:,bp_rp_cols] # extract the corresponding column

        f_bp_rps = a_vals[0] + a_vals[1]*bp_rps + a_vals[2]*bp_rps**2 + a_vals[3]*bp_rps**3 # construct the polynomials with the correct columns
        bre_corr = bres - f_bp_rps # compute the corrected BP-RP excess factor

        ## COMPUTE THE UNCERTAINTY OF THE CORRECTED BP-RP EXCESS FACTOR
        ## THIS IS DONE WITH A LINEAR EQUATION
        c0, c1, m = 0.0059898, 8.817481e-12, 7.618399
        sigma_c = c0 + c1 * apparent_mags**m

        ## TAKE THE CUT (Halbwachs+(2023))
        keep[np.abs(bre_corr) > 1.645*sigma_c] = False
    else:
        print("WARNING: phot_bp_rp_excess_factor, bp_rp, or phot_g_mean_mag not in table, skipping this cut.")
    
    return keep

### RESAMPLE COMPUTED TABLES ###
def resample_table(table, param, resample_func=None, resample_kwargs=None, n=None):
    """Resample a table by selecting a subset of rows based on a parameter.

    Parameters:
        table: (Table) The input table to resample.
        param: (str) The column name to use for resampling.
        resample_func: (callable) The function to use for resampling. Default is np.random.choice.
        resample_kwargs: (dict) Keyword arguments to pass to the resample function. Default is an empty dictionary.
        n: (int) The number of rows to select. Default is the length of the sorted values.

    Returns:
        (Table) The resampled table.
    """
    # default is to have the same number of objects
    if n is None:
            n = len(table)
    # in case no resample_kwargs are provided, set it to an empty dictionary
    if resample_kwargs is None:
        resample_kwargs = {}
    # by default, literally just resample the table
    if resample_func is None:
        return np.random.choice(table, size=n, **resample_kwargs)
    
    # now, operate normally.
    # the idea will be to sort the values of the parameter
    # then the supplied function will generate the values we want
    # and then we'll just look up the nearest table entry to each generated value
    values = table[param]
    sort_ids = np.argsort(values)
    sorted_values = values[sort_ids]
    
    # generate the values we want to sample
    generated_values = resample_func(n=n, **resample_kwargs)
    
    # find the nearest value in the sorted_values for each generated value
    idx = np.searchsorted(sorted_values, generated_values)
    # Clip to valid bounds
    idx = np.clip(idx, 1, len(sorted_values) - 1)

    left = sorted_values[idx - 1]
    right = sorted_values[idx]
    # Pick the nearest neighbor
    idx -= (generated_values - left) < (right - generated_values)
    
    # and return the table entries corresponding to those indices
    return table[sort_ids[idx]]