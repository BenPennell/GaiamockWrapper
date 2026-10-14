import numpy as np
import sys
import json
from pathlib import Path

with open('./config.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
folder_a_path = Path(d["gaiamock_path"])

if str(folder_a_path) not in sys.path:
    sys.path.append(str(folder_a_path))

'''
    We will use gaiamock_mod which handles each CCD as seperate
    
    This gives a more realistic RUWE distribution
'''

import gaiamock_mod # type: ignore
import gaiamock #type: ignore

SQRT8 = np.sqrt(8)

# Numeric payload order used by the astrometry callbacks. Keep this order in
# sync with the catalogue projection in SimulateAstrometry.py.
ASTROMETRY_FIELDS = (
    "ra", "dec", "pmra", "pmdec", "parallax", "phot_g_mean_mag",
    "mass", "m2", "period", "Tp", "ecc", "omega", "inc", "w", "f",
)

def generate_cfuncs():
    return gaiamock_mod.read_in_C_functions()


##################################
### --- COMPUTE ASTROMETRY --- ###
##################################

### --- ###
def astrometry_binary_contribution(parallax, m1, m2, period, Tp, ecc, omega, inc, w, f, t_ast_yr, psi, c_funcs):
    """
    This is the binary contribution to the observed motion as computed in predict_astrometry_luminous_binary() in gaiamock
    """
    
    EE = gaiamock_mod.solve_kepler_eqn_on_array(M = 2*np.pi/period * (t_ast_yr*365.25 - Tp), ecc = ecc, c_funcs = c_funcs)
    a_mas = gaiamock_mod.get_a_mas(period, m1, m2, parallax)
    A_pred = a_mas*( np.cos(w)*np.cos(omega) - np.sin(w)*np.sin(omega)*np.cos(inc) )
    B_pred = a_mas*( np.cos(w)*np.sin(omega) + np.sin(w)*np.cos(omega)*np.cos(inc) )
    F_pred = -a_mas*( np.sin(w)*np.cos(omega) + np.cos(w)*np.sin(omega)*np.cos(inc) )
    G_pred = -a_mas*( np.sin(w)*np.sin(omega) - np.cos(w)*np.cos(omega)*np.cos(inc) )
    
    X = np.cos(EE) - ecc
    Y = np.sqrt(1-ecc**2)*np.sin(EE)
    
    x, y = B_pred*X + G_pred*Y, A_pred*X + F_pred*Y   
    delta_eta = (-y*np.cos(psi) - x*np.sin(psi)) 
    bias = np.array([gaiamock_mod.al_bias_binary(delta_eta = delta_eta[i], q=m2/m1, f=f) for i in range(len(psi))])
    return bias

### --- ###
def compute_astrometry(ra, dec, pmra, pmdec, parallax, phot_g_mean_mag, c_funcs, data_release="dr3", reject_10_percent=True, binary_params=None, mod=True):  
    """
    Simulate the astrometry just as in gaiamock and gaiamock_mod
    """
    
    # get scanning times
    if mod:
        t = gaiamock_mod.get_gost_one_position(ra, dec, data_release=data_release)
    else:
        t = gaiamock.get_gost_one_position(ra, dec, data_release=data_release)
        
    # reject a random 10%
    if reject_10_percent:
        t = t[np.random.uniform(0, 1, len(t)) > 0.1]
        
    # get scanning angles, parallax factor, scan times (days)
    psi, plx_factor, jds = gaiamock_mod.fetch_table_element(['scanAngle[rad]', 'parallaxFactorAlongScan', 'ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]'], t)
    
    # rescale scanning times to years
    t_ast_yr = gaiamock_mod.rescale_times_astrometry(jd = jds, data_release = data_release)
    
    # we need both sources of noise
    # realistic noise applies the `real` noise to the observation
    # while the expected noise is the noise gaia `uses` for the fits, which isn't entirely correct
    if mod:
        epoch_err_per_transit = gaiamock_mod.get_realistic_epoch_astrometry_errors(ra, dec, phot_g_mean_mag)
        epoch_err_per_transit_expect = gaiamock_mod.al_uncertainty_per_ccd_interp(G = phot_g_mean_mag)
    else:    
        epoch_err_per_transit = gaiamock.al_uncertainty_per_ccd_interp(G = phot_g_mean_mag)/SQRT8 # non-mod case treats all CCDs as one
        epoch_err_per_transit_expect = epoch_err_per_transit
    
    # compute the single-star astrometry
    Lambda_pred = pmra*t_ast_yr*np.sin(psi) + pmdec*t_ast_yr*np.cos(psi) + parallax*plx_factor 

    if binary_params is not None: # for binaries, add the binary motion bias to the astrometry
        m1, m2, period, Tp, ecc, omega, inc, w, f = binary_params
        # too high of an eccentricity causes issues in al_bias_binary() where 'deta' never gets assigned
        ecc = np.minimum(ecc, 0.99)
        Lambda_pred += astrometry_binary_contribution(parallax, m1, m2, period, Tp, ecc, omega, inc, w, f, t_ast_yr, psi, c_funcs)
    
    # inflate with uncertainties
    Lambda_pred += epoch_err_per_transit*np.random.randn(len(psi)) # modeled noise
    
    if phot_g_mean_mag < 13 and not mod:
        extra_noise = np.random.uniform(0, 0.04)
    else: 
        extra_noise = 0
    Lambda_pred += extra_noise*np.random.randn(len(psi)) # unmodeled noise
    
    return t_ast_yr, psi, plx_factor, Lambda_pred, epoch_err_per_transit_expect*np.ones(len(Lambda_pred))

### --- ###
def astrometry_wrapper(source_values, c_funcs, is_binary, mod=True, **kwargs):
    """
    Wrapper for compute_astrometry() so that it's easier to interface from SimulateAstrometry.simulate_population_astrometry()
    """
    (ra, dec, pmra, pmdec, parallax, phot_g_mean_mag,
        mass, m2, period, Tp, ecc, omega, inc, w, f,) = source_values
    binary_params = None
    if is_binary:
        binary_params = [mass, m2, period, Tp, ecc, omega, inc, w, f]
    return compute_astrometry(ra=ra, dec=dec, parallax=parallax,
                                pmra=pmra, pmdec=pmdec, phot_g_mean_mag=phot_g_mean_mag,
                                binary_params=binary_params, mod=mod, c_funcs=c_funcs, **kwargs)

### --- ###
def check_ruwe(t_ast_yr, psi, plx_factor, ast_obs, ast_err, mod=True, return_fits=False):
    """
    Computes RUWE just as in gaiamock, this is directly pulled from there but modiefied for our situation
    """
    
    M = np.vstack([np.sin(psi), t_ast_yr*np.sin(psi), np.cos(psi), t_ast_yr*np.cos(psi), plx_factor]).T 
    Cinv = np.diag(1/ast_err**2)    
    mu = np.linalg.solve(M.T @ Cinv @ M, M.T @ Cinv @ ast_obs)  
    Lambda_solve = np.dot(M, mu)
    resids = ast_obs - Lambda_solve
    chi2_red_binned = np.sum(resids**2/ast_err**2)/(len(ast_obs) - 5)

    Nobs, nu, nu_unbinned = len(ast_obs), len(ast_obs) - 5, len(ast_obs)*8 - 5  
    # if we need to unbin things (using base gaiamock_mod)
    if mod:
        ruwe = np.sqrt(chi2_red_binned)
        if return_fits:
            cc = np.sqrt(chi2_red_binned/((1-2/(9*nu))**3 ))
            cov_matrix = np.linalg.inv(M.T @ Cinv @ M)
            sigma_mu = cc*np.sqrt(np.diag(cov_matrix))
            return ruwe, mu, sigma_mu
        return ruwe
    else:
        chi2_red_unbinned = gaiamock.predict_reduced_chi2_unbinned_data(chi2_red_binned = chi2_red_binned, n_param = 5, N_points = Nobs, Nbin=8)
        ruwe = np.sqrt(chi2_red_unbinned)
        if return_fits:
            cc = np.sqrt(chi2_red_unbinned/((1-2/(9*nu_unbinned))**3 ))
            cov_matrix = np.linalg.inv(M.T @ Cinv @ M)
            sigma_mu = cc*np.sqrt(np.diag(cov_matrix))
            return ruwe, mu, sigma_mu
        return ruwe  

################################################
### --- THE FULL DR3 ASTROMETRIC CASCADE --- ###
################################################

### --- ###
def return_values(soltype, mus, sis, stats=None, return_fits=False, return_statistics=False):
    return_mus = np.full(12, np.nan)
    return_mus[:len(mus)] = mus
    return_sis = np.full(12, np.nan)
    return_sis[:len(sis)] = sis
    if return_fits and return_statistics:
        return soltype, return_mus, return_sis, stats
    elif return_fits:
        return soltype, return_mus, return_sis
    elif return_statistics:
        return soltype, stats
    else:
        return soltype

def compute_ruwes(source_values, c_funcs, is_binary, mod=True, return_fits=False, **kwargs):
    """
    This computes the astrometry and then the 5p-model for the source
    """
    
    t_ast_yr, psi, plx_factor, ast_obs, ast_err = astrometry_wrapper(source_values, mod=mod, c_funcs=c_funcs, is_binary=is_binary, **kwargs)
    return check_ruwe(t_ast_yr, psi, plx_factor, ast_obs, ast_err, mod=mod, return_fits=return_fits)

### --- ###
def dr3_astrometric_pipeline(source_values, c_funcs, is_binary, skip_full=False, return_fits=False, return_statistics=False, **kwargs):
    """
    This function takes in a source and computes the astrometry, then checks the RUWE and determines the solution type.
    It returns the solution type, and optionally the RUWE and other fit parameters.
    The strategy here is to run it all in one go: compute the astrometry and report the fit/summary statistics.
    It turns out averaging over the CCDs (base gaiamock) is good enough for the higher-order solutions, but the
    gaiamock_mod is needed to get a more realistic RUWE distribution, so we use that for the first step, but 
    switch to base gaiamock for the higher-order solutions.
    
    This is to be run through SimulateAstrometry.simulate_population_astrometry()
    
    Parameters:
    - source_values: numeric sequence in ASTROMETRY_FIELDS order
    - t: the observation times (in years)
    - t_mod: the observation times for the modulated case (in years)
    - c_funcs: the C functions for the astrometry   
    
    Returns:
    - soltype: the solution type (0, 5, 7, 9, 12)
    - mu: the linear parameters (if return_fits is True), always a list of length 12
    - sigma_mu: the uncertainties on the linear parameters (if return_fits is True), always a list of length 12
    - summary_statistics: ruwe, and then the F2, s, parallax_over_error, sig_ecc for the 12 parameter solution if it was computed
    Both mu, sigma_mu will be front-filled with the corresponding number of parameters for the solution type
    """

    return_choices = {"return_fits": return_fits, "return_statistics": return_statistics}
    
    # COMPUTE ASTROMETRY, with gaiamock_mod, so use t_mod
    t_ast_yr, psi, plx_factor, ast_obs, ast_err = astrometry_wrapper(source_values, mod=True, c_funcs=c_funcs, is_binary=is_binary, **kwargs)
    
    # CHECK RUWE    
    res = check_ruwe(t_ast_yr, psi, plx_factor, ast_obs, ast_err, return_fits=return_fits, mod=True)
    if return_fits:
        ruwe, mu5, si5 = res
    else:
        ruwe = res
        mu5, si5 = [], []
    
    # SOLTYPE 0
    if ruwe < 1.4:
        return return_values(0, mu5, si5, stats={"ruwe": ruwe}, **return_choices)
    
    # HIGHER ORDER SOLUTIONS - USE NORMAL GAIAMOCK
    t_ast_yr, psi, plx_factor, ast_obs, ast_err = astrometry_wrapper(source_values, mod=False, c_funcs=c_funcs, is_binary=is_binary, **kwargs)
    
    # NVISIBILITY PERIODS
    N_visibility_periods = int(np.sum( np.diff(t_ast_yr*365.25) > 4) + 1)
    if (N_visibility_periods < 12) or (len(ast_obs) < 13): 
        return return_values(5, mu5, si5, stats={"ruwe": ruwe, "cut": "Nobs"}, **return_choices)
    
    # SOLTYPE 9
    F2_9par, s_9par, mu, sigma_mu = gaiamock_mod.check_9par(t_ast_yr, psi, plx_factor, ast_obs, ast_err)
    plx_over_err9 = mu[-1]/sigma_mu[-1]
    if (F2_9par < 25) and (s_9par > 12) and (plx_over_err9 > 2.1*s_9par**1.05):
        # some get sent back to 5 parameters
        if s_9par < 20:
            return return_values(5, mu5, si5, stats={"ruwe": ruwe, "F2": F2_9par, "s": s_9par, "parallax_over_error": plx_over_err9, "cut": "jerk"}, **return_choices)
        
        # otherwise, return 9 parameters
        return return_values(9, mu, sigma_mu, stats={"ruwe": ruwe, "F2": F2_9par, "s": s_9par, "parallax_over_error": plx_over_err9}, **return_choices)
    
    # SOLTYPE 7
    F2_7par, s_7par, mu, sigma_mu = gaiamock_mod.check_7par(t_ast_yr, psi, plx_factor, ast_obs, ast_err)
    plx_over_err7 = mu[-1]/sigma_mu[-1]
    if (F2_7par < 25) and (s_7par > 12) and (plx_over_err7 > 1.2*s_7par**1.05):
        # some get sent back to 5 parameters
        if (s_7par < 20) or (F2_7par > 22):
            return return_values(5, mu5, si5, stats={"ruwe": ruwe, "F2": F2_7par, "s": s_7par, "parallax_over_error": plx_over_err7, "cut": "acceleration"}, **return_choices)
        
        # otherwise, return 7 parameters
        return return_values(7, mu, sigma_mu, stats={"ruwe": ruwe, "F2": F2_7par, "s": s_7par, "parallax_over_error": plx_over_err7}, **return_choices)
    
    # IF YOU DON'T EVEN WANT TO TRY
    if skip_full:
        return return_values(5, mu5, si5, stats={"ruwe": ruwe, "cut": "skip full"}, **return_choices)
    
    # DON'T EVEN BOTHER AT HIGH P
    if is_binary:
        if source_values[8] > 1e4:
            return return_values(5, mu5, si5, stats={"ruwe": ruwe, "cut": "high period"}, **return_choices)
    
    # SOLTYPE 12
    res = gaiamock.fit_orbital_solution_nonlinear(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, c_funcs = c_funcs, L = np.array([10, 0, 0]))
    
    # get the linear parameters 
    period, phi_p, ecc = res
    chi2, mu_linear = gaiamock.get_astrometric_chi2(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, P = period, phi_p = phi_p, ecc = ecc, c_funcs=c_funcs)
    ra_off, pmra, dec_off, pmdec, plx, B, G, A, F = mu_linear
    p0 = [ra_off, dec_off, plx, pmra, pmdec, period, ecc, phi_p, A, B, F, G]
    
    errors, a0_mas, sigma_a0_mas, inc_deg = gaiamock.get_uncertainties_at_best_fit_binary_solution(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, p0 = p0, c_funcs = c_funcs)
    sig_parallax, sig_ecc = errors[2], errors[6]
    Nobs, nu, nu_unbinned = len(ast_obs), len(ast_obs) - 12, len(ast_obs)*8 - 12
    chi2_red = chi2/nu
    
    F2 = gaiamock.predict_F2_unbinned_data(chi2_red_binned = chi2_red, n_param = 12, N_points = Nobs, Nbin=8)
    a0_over_err, parallax_over_error = a0_mas/sigma_a0_mas, plx/sig_parallax

    if (F2 < 25) and (a0_over_err > 158/np.sqrt(period)) and (a0_over_err > 5) and (parallax_over_error > 20000/period) and (sig_ecc < 0.079*np.log(period)-0.244):
        return return_values(12, p0, errors, stats={"ruwe": ruwe, "F2": F2, "s": a0_over_err, "parallax_over_error": parallax_over_error, "sig_ecc": sig_ecc}, **return_choices)
    
    # SOLTYPE 5 - if nothing else worked
    return return_values(5, mu5, si5, stats={"ruwe": ruwe, "F2": F2, "s": a0_over_err, "parallax_over_error": parallax_over_error, "sig_ecc": sig_ecc, "cut": "orbit"}, **return_choices)

### --- ###
def dr3_astrometric_cascade(t_ast_yr, psi, plx_factor, ast_obs, ast_err, c_funcs, skip_full=False, return_fits=False, return_statistics=False, period=None):
    """
    A copy of dr3_astrometric_pipeline, but instead of computing the astrometry from source_values, it takes in the astrometry directly.
    This cannot do the time save of only using gaiamock_mod to compute the RUWE, so be cautious.
    To be run through SimulateAstrometry.run_astrometric_cascade()
    
    Parameters:
    - the astrometry spit out of gaiamock or the stuff I have
    - c_funcs: the C functions for the astrometry   
    
    Returns:
    - soltype: the solution type (0, 5, 7, 9, 12)
    - mu: the linear parameters (if return_fits is True), always a list of length 12
    - sigma_mu: the uncertainties on the linear parameters (if return_fits is True), always a list of length 12
    - summary_statistics: ruwe, and then the F2, s, parallax_over_error, sig_ecc for the 12 parameter solution if it was computed
    Both mu, sigma_mu will be front-filled with the corresponding number of parameters for the solution type
    """

    return_choices = {"return_fits": return_fits, "return_statistics": return_statistics}
    
    # CHECK RUWE    
    res = check_ruwe(t_ast_yr, psi, plx_factor, ast_obs, ast_err, return_fits=return_fits, mod=True)
    if return_fits:
        ruwe, mu5, si5 = res
    else:
        ruwe = res
        mu5, si5 = [], []
    
    # SOLTYPE 0
    if ruwe < 1.4:
        return return_values(0, mu5, si5, stats={"ruwe": ruwe}, **return_choices)
    
    # NVISIBILITY PERIODS
    N_visibility_periods = int(np.sum( np.diff(t_ast_yr*365.25) > 4) + 1)
    if (N_visibility_periods < 12) or (len(ast_obs) < 13): 
        return return_values(5, mu5, si5, stats={"ruwe": ruwe, "cut": "Nobs"}, **return_choices)
    
    # SOLTYPE 9
    F2_9par, s_9par, mu, sigma_mu = gaiamock_mod.check_9par(t_ast_yr, psi, plx_factor, ast_obs, ast_err)
    plx_over_err9 = mu[-1]/sigma_mu[-1]
    if (F2_9par < 25) and (s_9par > 12) and (plx_over_err9 > 2.1*s_9par**1.05):
        # some get sent back to 5 parameters
        if s_9par < 20:
            return return_values(5, mu5, si5, stats={"ruwe": ruwe, "F2": F2_9par, "s": s_9par, "parallax_over_error": plx_over_err9, "cut": "jerk"}, **return_choices)
        
        # otherwise, return 9 parameters
        return return_values(9, mu, sigma_mu, stats={"ruwe": ruwe, "F2": F2_9par, "s": s_9par, "parallax_over_error": plx_over_err9}, **return_choices)
    
    # SOLTYPE 7
    F2_7par, s_7par, mu, sigma_mu = gaiamock_mod.check_7par(t_ast_yr, psi, plx_factor, ast_obs, ast_err)
    plx_over_err7 = mu[-1]/sigma_mu[-1]
    if (F2_7par < 25) and (s_7par > 12) and (plx_over_err7 > 1.2*s_7par**1.05):
        # some get sent back to 5 parameters
        if (s_7par < 20) or (F2_7par > 22):
            return return_values(5, mu5, si5, stats={"ruwe": ruwe, "F2": F2_7par, "s": s_7par, "parallax_over_error": plx_over_err7, "cut": "acceleration"}, **return_choices)
        
        # otherwise, return 7 parameters
        return return_values(7, mu, sigma_mu, stats={"ruwe": ruwe, "F2": F2_7par, "s": s_7par, "parallax_over_error": plx_over_err7}, **return_choices)
    
    # IF YOU DON'T EVEN WANT TO TRY
    if skip_full:
        return return_values(5, mu5, si5, stats={"ruwe": ruwe, "cut": "skip full"}, **return_choices)
    
    # DON'T EVEN BOTHER AT HIGH P
    if period is not None:
        if period > 1e4:
            return return_values(5, mu5, si5, stats={"ruwe": ruwe, "cut": "high period"}, **return_choices)
    
    # SOLTYPE 12
    res = gaiamock.fit_orbital_solution_nonlinear(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, c_funcs = c_funcs, L = np.array([10, 0, 0]))
    
    # get the linear parameters 
    period, phi_p, ecc = res
    chi2, mu_linear = gaiamock.get_astrometric_chi2(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, P = period, phi_p = phi_p, ecc = ecc, c_funcs=c_funcs)
    ra_off, pmra, dec_off, pmdec, plx, B, G, A, F = mu_linear
    p0 = [ra_off, dec_off, plx, pmra, pmdec, period, ecc, phi_p, A, B, F, G]
    
    errors, a0_mas, sigma_a0_mas, inc_deg = gaiamock.get_uncertainties_at_best_fit_binary_solution(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, p0 = p0, c_funcs = c_funcs)
    sig_parallax, sig_ecc = errors[2], errors[6]
    Nobs, nu, nu_unbinned = len(ast_obs), len(ast_obs) - 12, len(ast_obs)*8 - 12
    chi2_red = chi2/nu
    
    F2 = gaiamock.predict_F2_unbinned_data(chi2_red_binned = chi2_red, n_param = 12, N_points = Nobs, Nbin=8)
    a0_over_err, parallax_over_error = a0_mas/sigma_a0_mas, plx/sig_parallax

    if (F2 < 25) and (a0_over_err > 158/np.sqrt(period)) and (a0_over_err > 5) and (parallax_over_error > 20000/period) and (sig_ecc < 0.079*np.log(period)-0.244):
        return return_values(12, p0, errors, stats={"ruwe": ruwe, "F2": F2, "s": a0_over_err, "parallax_over_error": parallax_over_error, "sig_ecc": sig_ecc}, **return_choices)
    
    # SOLTYPE 5 - if nothing else worked
    return return_values(5, mu5, si5, stats={"ruwe": ruwe, "F2": F2, "s": a0_over_err, "parallax_over_error": parallax_over_error, "sig_ecc": sig_ecc, "cut": "orbit"}, **return_choices)

def dr3_astrometric_pipeline_direct(source_values, c_funcs, is_binary, skip_full=False, return_fits=False, return_statistics=False, **kwargs):
    """
    Computes the astrmoetry and assigns the solution type in the DR3 regime but with only one invocation of gaiamock
    """
    
    t_ast_yr, psi, plx_factor, ast_obs, ast_err = astrometry_wrapper(source_values, c_funcs=c_funcs, is_binary=is_binary, **kwargs)
    return dr3_astrometric_cascade(t_ast_yr, psi, plx_factor, ast_obs, ast_err, c_funcs, skip_full=skip_full, return_fits=return_fits, return_statistics=return_statistics, period=source_values[8])

################################################
### --- THE FULL DR4 ASTROMETRIC CASCADE --- ###
################################################

# this is temporary, to be replaced with the actual DR4 astrometric pipeline once it is available

def dr4_astrometric_cascade(t_ast_yr, psi, plx_factor, ast_obs, ast_err, c_funcs, return_fits=False, return_statistics=False, period=None):
    """
        This is the "Gaia BH3 cuts" guess, where I only fit 5 or 12 parameters.
    """  
    res = gaiamock.fit_orbital_solution_nonlinear(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, c_funcs = c_funcs, L = np.array([10, 0, 0]))
    ruwe = check_ruwe(t_ast_yr, psi, plx_factor, ast_obs, ast_err, return_fits=False, mod=False)
    
    # get the linear parameters 
    period, phi_p, ecc = res
    chi2, mu_linear = gaiamock.get_astrometric_chi2(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, P = period, phi_p = phi_p, ecc = ecc, c_funcs=c_funcs)
    ra_off, pmra, dec_off, pmdec, plx, B, G, A, F = mu_linear
    p0 = [ra_off, dec_off, plx, pmra, pmdec, period, ecc, phi_p, A, B, F, G]
    
    errors, a0_mas, sigma_a0_mas, inc_deg = gaiamock.get_uncertainties_at_best_fit_binary_solution(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, p0 = p0, c_funcs = c_funcs)
    sig_parallax, sig_ecc = errors[2], errors[6]
    Nobs, nu, nu_unbinned = len(ast_obs), len(ast_obs) - 12, len(ast_obs)*8 - 12
    chi2_red = chi2/nu
    
    F2 = gaiamock.predict_F2_unbinned_data(chi2_red_binned = chi2_red, n_param = 12, N_points = Nobs, Nbin=8)
    a0_over_err, parallax_over_error = a0_mas/sigma_a0_mas, plx/sig_parallax

    soltype = 5
    # check the BH3 cuts
    if (F2 < 25) and (a0_over_err > 158/np.sqrt(period)) and (a0_over_err > 5) and (parallax_over_error > 20000/period) and (sig_ecc < 0.079*np.log(period)-0.244):
        soltype = 12    
    
    return return_values(soltype, p0, errors, stats = {"ruwe": ruwe, "F2": F2, "a0_over_err": a0_over_err, "parallax_over_error": parallax_over_error, "sig_ecc": sig_ecc}, 
                         return_fits=return_fits, return_statistics=return_statistics)

def dr4_astrometric_pipeline(source_values, c_funcs, is_binary, mod=True, return_fits=False, return_statistics=False, **kwargs):
    """
    Computes the astrometry and assigns the solution type based on Gaia BH3 cuts
    """
    t_ast_yr, psi, plx_factor, ast_obs, ast_err = astrometry_wrapper(source_values, mod=mod, c_funcs=c_funcs, is_binary=is_binary, **kwargs)
    return dr4_astrometric_cascade(t_ast_yr, psi, plx_factor, ast_obs, ast_err, c_funcs, return_fits=return_fits, return_statistics=return_statistics, period=source_values[8])