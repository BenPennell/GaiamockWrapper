import numpy as np

### --- ###
def calculate_orbit_parameter(m, q, w):
    """
        This is lambda
    """
    return q*w*m**(1/3)*(1 + q)**(-2/3)

### --- ###
def q_from_l(l, m, w):
    """
        sort of a nightmare to disentangle the nonlinear q dependence
        in lambda, this function solves it numerically if that's ever
        needed
    """
    z = m * (w / l)**3
    # Coefficients of z q^3 - q^2 - 2q - 1 = 0
    coeff = [z, -1.0, -2.0, -1.0]
    roots = np.roots(coeff)
    # real roots only
    real_roots = roots[np.isreal(roots)].real
    # choose the physically valid one: q > 0
    valid = real_roots[real_roots > 0]
    if len(valid) == 0:
        return -1.0
    # Usually only one positive root exists
    return valid[0]

### --- ###
def q_from_l_vectorized(l_array, m, w):
    """
        vectorised version of q_from_l()
    """
    z = m * (w / l_array)**3

    # coefficients for all cubics
    coeffs = np.column_stack([z, -np.ones_like(z), -2*np.ones_like(z), -1*np.ones_like(z)])
    roots = np.array([np.roots(c) for c in coeffs])  # shape (N, 3)

    # real roots mask
    real_roots = roots.real * np.isreal(roots)  # imaginary parts removed

    # positive roots mask
    positive_mask = real_roots > 0

    # pick the first positive root (there should be exactly one)
    q_vals = np.where(positive_mask.any(axis=1),
                      real_roots[np.arange(len(real_roots)), positive_mask.argmax(axis=1)],
                      -1.0)
    return q_vals

### --- ###
def scale_resolution(arr, scale=2, axis=0, even=False):
    """
        upscales grid resolution horizontally by splitting grid values evenly into multiple cells
    """
    # Create a new shape with double the size along the specified axis
    new_shape = list(arr.shape)
    new_shape[axis] *= scale

    # Expand the array along a new axis after the target one
    expanded = np.expand_dims(arr, axis + 1)  # shape becomes (..., 1, ...)
    
    # Repeat the values along the new axis (splitting them evenly)
    repeated = np.repeat(expanded, scale, axis=axis + 1)
    if even:
        repeated = repeated / scale

    # Reshape back by merging the expanded axis with the original one
    transposed = np.reshape(repeated, new_shape)

    return transposed

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

### --- ###
def convert_binarity(fb, a):
    """
        convert from binary fraction within some range to a total binary fraction
        where fb is the fraction of binaries in the range of interest,
        a is the fraction of all binaries that fall within that range (i.e. the area under the distribution in that range)
    """
    return a / (a + 1/fb - 1)

### --- ###
def convert_to_fb(f, p_model, pcut=8, resolution=100):
    """
        convert from a total binary fraction to the binary fraction within some range of interest, 
        where f is the total binary fraction, 
        p_model is the period distribution model, 
        and pcut is the upper limit of the period
    """
    a = cutoff_to_fraction(p_model, pcut, resolution=resolution)
    return convert_binarity(f, a)

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
def generate_parallax(dist_range=(100,200), resolution=1000):
    # sample a distance from a distribution that goes as d^2 between 100 and 200
    # from inverse cdf sampling
    dists = np.linspace(*dist_range, resolution)
    d_pdf = np.zeros_like(dists)
    d_pdf[1:] = pexp(dists[1:], 2) # d^2
    d_cdf = np.cumsum(d_pdf / np.sum(d_pdf))
    d = np.interp(np.random.rand(), d_cdf, dists)
    parallax = 1000 / d 
    return parallax

def relative_volume(mag, plx, dist_range=(100,200), resolution=1000):
    new_plx = adjust_parallax(plx, mag, 17.65) # see where the XP cutoff is
    if new_plx < 1000/dist_range[1]:
        return 1 # if we don't cut it off, the effective volume is 1
    
    covered_area = area_in_range_powerlaw((dist_range[0],1000/new_plx), 2, resolution=resolution)
    total_area = area_in_range_powerlaw(dist_range, 2, resolution=resolution)
    
    return covered_area/total_area # otherwise, return the fractional area where we can see the object
    