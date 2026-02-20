import numpy as np
import sys
import os
import json

with open('config.json') as f:
    d = json.load(f)
    
folder_a_path = os.path.abspath(os.path.join(os.getcwd(), d["gaiamock_path"]))
if folder_a_path not in sys.path:
    sys.path.append(folder_a_path)

import gaiamock

def generate_cfuncs():
    return gaiamock.read_in_C_functions()

## FUNCTION TO DETERMINE SOLUTION TYPE
def predict_astrometry_luminous_binary(ra, dec, parallax, pmra, pmdec, m1, m2, period, Tp, ecc, omega, inc, w, phot_g_mean_mag, f, t, data_release, c_funcs, reject_10_percent = True):
    '''
    reduced version of the gaiamock one
    '''
    # too high of an eccentricity causes issues in al_bias_binary() where 'deta' never gets assigned
    ecc = np.minimum(ecc, 0.99)
    # t = get_gost_one_position(ra, dec, data_release=data_release)
    # reject a random 10%
    if reject_10_percent:
        t = t[np.random.uniform(0, 1, len(t)) > 0.1]
    psi, plx_factor, jds = gaiamock.fetch_table_element(['scanAngle[rad]', 'parallaxFactorAlongScan', 'ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]'], t)
    t_ast_yr = gaiamock.rescale_times_astrometry(jd = jds, data_release = data_release)
    
    epoch_err_per_transit = gaiamock.get_realistic_epoch_astrometry_errors(ra, dec, phot_g_mean_mag)
    epoch_err_per_transit_expect = gaiamock.al_uncertainty_per_ccd_interp(G = phot_g_mean_mag)

    EE = gaiamock.solve_kepler_eqn_on_array(M = 2*np.pi/period * (t_ast_yr*365.25 - Tp), ecc = ecc, c_funcs = c_funcs)
    a_mas = gaiamock.get_a_mas(period, m1, m2, parallax)
    A_pred = a_mas*( np.cos(w)*np.cos(omega) - np.sin(w)*np.sin(omega)*np.cos(inc) )
    B_pred = a_mas*( np.cos(w)*np.sin(omega) + np.sin(w)*np.cos(omega)*np.cos(inc) )
    F_pred = -a_mas*( np.sin(w)*np.cos(omega) + np.cos(w)*np.sin(omega)*np.cos(inc) )
    G_pred = -a_mas*( np.sin(w)*np.sin(omega) - np.cos(w)*np.cos(omega)*np.cos(inc) )
    cpsi, spsi = np.cos(psi), np.sin(psi)
    
    X = np.cos(EE) - ecc
    Y = np.sqrt(1-ecc**2)*np.sin(EE)
    
    x, y = B_pred*X + G_pred*Y, A_pred*X + F_pred*Y   
    delta_eta = (-y*cpsi - x*spsi) 
    bias = np.array([gaiamock.al_bias_binary(delta_eta = delta_eta[i], q=m2/m1, f=f) for i in range(len(psi))])
    Lambda_com = pmra*t_ast_yr*spsi + pmdec*t_ast_yr*cpsi + parallax*plx_factor # barycenter motion
    Lambda_pred = Lambda_com + bias # binary motion

    Lambda_pred += epoch_err_per_transit*np.random.randn(len(psi)) # modeled noise
    
    return t_ast_yr, psi, plx_factor, Lambda_pred, epoch_err_per_transit_expect*np.ones(len(Lambda_pred))

def rapid_solution_type(period, q, parallax, m1, phot_g_mean_mag, f, ecc, inc, w, omega, Tp, ra, dec, pmra, pmdec, t, c_funcs, skip_full=False):
    # COMPUTE ASTROMETRY
    t_ast_yr, psi, plx_factor, ast_obs, ast_err = predict_astrometry_luminous_binary(ra = ra, dec = dec, parallax = parallax, 
                    pmra = pmra, pmdec = pmdec, m1 = m1, m2 = q*m1, period = period, Tp = Tp*period, ecc = ecc, 
                    omega = omega, inc = inc, w = w, phot_g_mean_mag = phot_g_mean_mag, f = f, data_release = "dr3", t=t,
                    c_funcs = c_funcs)
    
    # CHECK RUWE
    Cinv = np.diag(1/ast_err**2)    
    M = np.vstack([np.sin(psi), t_ast_yr*np.sin(psi), np.cos(psi), t_ast_yr*np.cos(psi), plx_factor]).T 
    mu = np.linalg.solve(M.T @ Cinv @ M, M.T @ Cinv @ ast_obs)  
    Lambda_pred = np.dot(M, mu)
    resids = ast_obs - Lambda_pred
    Nobs, nu = len(ast_obs), len(ast_obs) - 5 
    chi2_red = np.sum(resids**2/ast_err**2)/nu

    ruwe = np.sqrt(chi2_red)
    # SOLTYPE 0
    if ruwe < 1.4:
        return 0
    
    # NVISIBILITY PERIODS
    N_visibility_periods = int(np.sum( np.diff(t_ast_yr*365.25) > 4) + 1)
    if (N_visibility_periods < 12) or (len(ast_obs) < 13): 
        return 5
    
    # SOLTYPE 9
    F2_9par, s_9par, mu, sigma_mu = gaiamock.check_9par(t_ast_yr, psi, plx_factor, ast_obs, ast_err)
    plx_over_err9 = mu[-1]/sigma_mu[-1]
    if (F2_9par < 25) and (s_9par > 12) and (plx_over_err9 > 2.1*s_9par**1.05):
        return 9
    
    # SOLTYPE 7
    F2_7par, s_7par, mu, sigma_mu = gaiamock.check_7par(t_ast_yr, psi, plx_factor, ast_obs, ast_err)
    plx_over_err7 = mu[-1]/sigma_mu[-1]
    if (F2_7par < 25) and (s_7par > 12) and (plx_over_err7 > 1.2*s_7par**1.05):
        return 7
    
    # IF YOU DON'T EVEN WANT TO TRY
    if skip_full:
        return 5
    
    # DON'T EVEN BOTHER AT HIGH P
    if period > 1e4:
        return 5
    
    # SOLTYPE 12
    res = gaiamock.fit_orbital_solution_nonlinear(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, c_funcs = c_funcs, L = np.array([10, 0, 0]))
    
    # get the linear parameters 
    period, phi_p, ecc = res
    chi2, mu_linear = gaiamock.get_astrometric_chi2(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, P = period, phi_p = phi_p, ecc = ecc, c_funcs=c_funcs)
    ra_off, pmra, dec_off, pmdec, plx, B, G, A, F = mu_linear
    p0 = [ra_off, dec_off, plx, pmra, pmdec, period, ecc, phi_p, A, B, F, G]
    
    errors, a0_mas, sigma_a0_mas, inc_deg = gaiamock.get_uncertainties_at_best_fit_binary_solution(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, p0 = p0, c_funcs = c_funcs)
    sig_parallax, sig_ecc = errors[2], errors[6]
    nu = len(ast_obs) - 12
    chi2_red = chi2/nu
    
    F2 = np.sqrt(9*nu/2)*(chi2_red**(1/3) + 2/(9*nu) - 1)
    a0_over_err, parallax_over_error = a0_mas/sigma_a0_mas, plx/sig_parallax

    if (F2 < 25) and (a0_over_err > 158/np.sqrt(period)) and (a0_over_err > 5) and (parallax_over_error > 20000/period) and (sig_ecc < 0.079*np.log(period)-0.244):
        return 12
    
    # SOLTYPE 5 - if nothing else worked
    return 5