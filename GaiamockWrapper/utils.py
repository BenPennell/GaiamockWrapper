from astropy.table import Table
import numpy as np

### DEFAULT RANDOM ORBITAL ANGLE GENERATORS
def random_angle(n=1):
    x = np.random.rand(n) * 2 * np.pi
    return float(x[0]) if n == 1 else x

def random_inc(n=1):
    x = np.arccos(2 * np.random.rand(n) - 1) # Flat in cos(i)
    return float(x[0]) if n == 1 else x

def random_Tp(n=1):
    x = np.random.rand(n) - 0.5
    return float(x[0]) if n == 1 else x

### RANDOM HELPER FUNCTIONS
### --- ###
def gaussian(x, mu, sigma):
    """
        Normalised gaussian at x, defined by two
        parameters: peak (mu) and width (sigma)
    """
    return np.exp(-(mu - x)**2/(2*sigma**2)) / np.sqrt(2 * np.pi * sigma**2)

### --- ###
def area_in_range(target_range, mu, sigma, resolution=100):
    xs = np.linspace(*target_range, resolution)
    ys = gaussian(xs, mu, sigma)
    return np.trapezoid(y=ys, x=xs)

### --- ###
def pexp(val, index, val_range=(0, 1), ignore_a=False):
    """
        normalised power law probability
    """
    a = 1
    if not ignore_a:
        a = (index + 1) / (val_range[1] ** (index + 1) - val_range[0] ** (index + 1))
    return a * (val ** index)

### -- ###
def area_in_range_powerlaw(target_range, index, resolution=100):
    xs = np.linspace(*target_range, resolution)
    ys = pexp(xs, index, ignore_a=True)
    return np.trapezoid(y=ys, x=xs)

### --- ###
def cutoff_to_fraction(p_model, pcut, resolution=100):
    p_mu, p_si = p_model
    total_area = area_in_range((1,pcut), p_mu, p_si, resolution)
    observable_area = area_in_range((2,3), p_mu, p_si, resolution)
    return observable_area / total_area

### --- ###
def fraction_to_cutoff(p_model, fraction, resolution=100):
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

### DEFAULT ORBITAL PARAMETER DISTRIBUTIONS
# you can follow how these are created when making your own
### --- ###
def choose_value(cdf, grid, size):
    """
        Helper function to sample from a distribution defined by a cdf and grid of values
    """
    u = np.random.uniform(cdf.min(), cdf.max(), size)
    return np.interp(u, cdf, grid)

def return_distribution(vals, log=False, logbase=10, **kwargs):
    """
        Helper function which processes the output of a distribution function to return either log or linear values
    """
    if log:
        vals = logbase ** vals
    return vals

# simple distributions
def constant_distribution(n=1, value=2.5, **kwargs):
    return np.full(n, value)

def uniform_distribution(n=1, lims=(2, 3), **kwargs):
    ys = np.random.uniform(lims[0], lims[1], n)

    return return_distribution(ys, **kwargs)

def normal_distribution(n=1, gaussian_params=(5, 3), lims=(2, 3), resolution=1000, **kwargs):
    mu, si = gaussian_params
    xs = np.linspace(*lims, resolution)
    pdf = gaussian(xs, mu, si)
    cdf = np.cumsum(pdf / pdf.sum())
    ys = choose_value(cdf, xs, n)

    return return_distribution(ys, **kwargs)

def power_law_distribution(n=1, index=2, lims=(2, 3), resolution=100, **kwargs):
    xs = np.linspace(*lims, resolution)
    pdf = pexp(xs, index, val_range=lims)
    cdf = np.cumsum(pdf / pdf.sum())
    ys = choose_value(cdf, xs, n)

    return return_distribution(ys, **kwargs)

def flux_ratio_power_law(index=4, mass_ratios=None, **kwargs):
    return mass_ratios ** index

### FUCNTIONS FOR RESCALING-SAMPLES TO DIFFERENT DISTANCES
### --- ###
def adjust_magnitude(mag, plx, new_plx):
    """Adjust the magnitude of a star to reflect a change in parallax."""
    # Calculate the absolute magnitude
    abs_mag = mag - 5 * np.log10(1000/plx) + 5
    
    # Calculate the new apparent magnitude
    new_mag = abs_mag + 5 * np.log10(1000/new_plx) - 5
    return new_mag

### --- ###
def adjust_parallax(plx, mag, new_mag):
    """Adjust the parallax of a star to reflect a change in magnitude."""
    # Calculate the absolute magnitude
    abs_mag = mag - 5 * np.log10(1000/plx) + 5
    
    # Calculate the new parallax
    new_plx = 10 * 10**((abs_mag - new_mag + 5) / 5)
    return new_plx

### --- ###
def generate_parallax(n=1, dist_range=(100,200), resolution=1000, **kwargs):
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
    soltypes = np.zeros(len(catalogue), dtype=int)
    for i, result in enumerate(astrometric_results):
        soltypes[i] = result[0]
    catalogue["solution_type"] = soltypes

### --- ###
def postprocess_fits(astrometric_results):
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
        