import numpy as np
import gaiamock.gaiamock as gaiamock
import pickle
import numbers
import sympy
from concurrent.futures import ProcessPoolExecutor

try:
    # for Jupyter
    from tqdm.notebook import tqdm
except ImportError:
    # for terminal
    from tqdm import tqdm

c_funcs = gaiamock.read_in_C_functions()


### --- BUILT-IN SAMPLING FUNCTIONS --- ###
'''
    Note that on top of this, randomly sampling from
    catalogue is available by setting the kwarg for that
    parameter to "catalogue" and providing a path to your
    catalogue through the keyword "catalogue"
'''

### --- ###
def sample_with_noise(value, scale):
    return np.random.normal(value, scale)

### --- ###
def random_angle():
    return np.random.rand()*2*np.pi

def random_inc():
    return np.random.rand()*0.5*np.pi

def random_Tp():
    return np.random.rand()-0.5 # generally you get Tp in (-0.5,0.5)*period

def sample_perfect(func, boundary, count, *func_params, resolution=1000):
    vals = np.linspace(*boundary, resolution)
    p_vals = func(vals, *func_params)
    pdf_vals = np.cumsum(p_vals)
    y_targets = np.linspace(0, 1, count)
    return np.array(vals[[np.argmin(abs(y - pdf_vals)) for y in y_targets]])

### --- BUILT-IN FUNCTIONS FOR MARGINALIZING --- ###

### --- ###
def astrometry_setup(ra, dec, phot_g_mean_mag, data_release):
    t = gaiamock.get_gost_one_position(ra, dec, data_release=data_release)
        
    # reject a random 10%
    t = t[np.random.uniform(0, 1, len(t)) > 0.1]
    psi, plx_factor, jds = gaiamock.fetch_table_element(['scanAngle[rad]', 'parallaxFactorAlongScan', 'ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]'], t)
    t_ast_yr = gaiamock.rescale_times_astrometry(jd = jds, data_release = data_release)
    
    N_ccd_avg = 8
    epoch_err_per_transit = gaiamock.al_uncertainty_per_ccd_interp(G = phot_g_mean_mag)/np.sqrt(N_ccd_avg)
    
    return t_ast_yr, psi, plx_factor, epoch_err_per_transit

### --- ###
def predict_astrometry_luminous_binary_reduced(t_ast_yr, psi, plx_factor, epoch_err_per_transit,
                                               phot_g_mean_mag, parallax, pmra, pmdec, m1, m2, period, Tp, ecc, omega, inc, w, f, c_funcs):
    '''
    simplified version of gaiamock.predict_astrometry_luminous_binary()
    '''
    
    if phot_g_mean_mag < 13:
        extra_noise = np.random.uniform(0, 0.04)
    else: 
        extra_noise = 0
    
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
    Lambda_pred += extra_noise*np.random.randn(len(psi)) # unmodeled noise
    
    return Lambda_pred, epoch_err_per_transit*np.ones(len(Lambda_pred))

### --- ###
def check_ruwe_reduced(t_ast_yr, psi, plx_factor, ast_obs, ast_err):
    '''
        reduced function for checking RUWE from gaiamock
    '''
    Cinv = np.diag(1/ast_err**2)    
    M = np.vstack([np.sin(psi), t_ast_yr*np.sin(psi), np.cos(psi), t_ast_yr*np.cos(psi), plx_factor]).T 
    mu = np.linalg.solve(M.T @ Cinv @ M, M.T @ Cinv @ ast_obs)  
    Lambda_pred = np.dot(M, mu)
    resids = ast_obs - Lambda_pred
    Nobs, nu, _ = len(ast_obs), len(ast_obs) - 5, len(ast_obs)*8 - 5  
    chi2_red_binned = np.sum(resids**2/ast_err**2)/nu
    chi2_red_unbinned = gaiamock.predict_reduced_chi2_unbinned_data(chi2_red_binned = chi2_red_binned, n_param = 5, N_points = Nobs, Nbin=8)
    
    ruwe = np.sqrt(chi2_red_unbinned)
    return ruwe

### --- ###
def calculate_ruwe(t_ast_yr=None, psi=None, plx_factor=None, epoch_err_per_transit=None, df=None,m1=0.4,q=0.2,period=1e4,ecc=0,inc=0,Tp=0,omega=0,w=0,f=1e-2,phot_g_mean_mag=14,parallax=None,ra=None,dec=None,pmra=0,pmdec=0,return_astrometry=False,return_both=False):
    if df is not None:
        if parallax is None:
            parallax = df["parallax"]
        if ra is None:
            ra = df["ra"]
        if dec is None:
            dec = df["dec"]
    ecc = np.minimum(ecc, 0.99) # there's a weird numerical problem if eccentricity is too high
    # this could also be fixed in gaiamock rescale_times_astrometry() by providing an else: case for
    # when you have an unphyisical eccentricity. but of course this is not mindful
    
    # if part of the astrometry is precomputed, do that
    if t_ast_yr is not None:
        ast_obs, ast_err = predict_astrometry_luminous_binary_reduced(t_ast_yr, psi, plx_factor, epoch_err_per_transit, 
                                                        parallax = parallax, pmra = pmra, pmdec = pmdec, 
                                                        m1 = m1, m2 = q*m1, period = period, Tp = Tp*period, ecc = ecc, 
                                                        omega = omega, inc = inc, w = w, f = f,
                                                        c_funcs = c_funcs, phot_g_mean_mag = phot_g_mean_mag)
        astrometry = (t_ast_yr, psi, plx_factor, ast_obs, ast_err)
    else:
        astrometry = gaiamock.predict_astrometry_luminous_binary(ra = ra, dec = dec, parallax = parallax, 
                        pmra = pmra, pmdec = pmdec, m1 = m1, m2 = q*m1, period = period, Tp = Tp*period, ecc = ecc, 
                        omega = omega, inc = inc, w = w, phot_g_mean_mag = phot_g_mean_mag, f = f, data_release = 'dr3',
                        c_funcs = c_funcs)
    if return_astrometry:
        return astrometry
    if return_both:
        return check_ruwe_reduced(*astrometry), astrometry
    return check_ruwe_reduced(*astrometry)

### --- ###
def calculate_ruwe_ss(df=None,phot_g_mean_mag=None,parallax=None,ra=None,dec=None,pmra=0,pmdec=0,return_astrometry=False,return_both=False):
    if df is not None:
        if phot_g_mean_mag is None:
            phot_g_mean_mag = df["phot_g_mean_mag"]
        if parallax is None:
            parallax = df["parallax"]
        if ra is None:
            ra = df["ra"]
        if dec is None:
            dec = df["dec"]
    astrometry = gaiamock.predict_astrometry_single_source(ra = ra, dec = dec, parallax = parallax, 
                        pmra = pmra, pmdec = pmdec, phot_g_mean_mag = phot_g_mean_mag, data_release = 'dr3',
                        c_funcs = c_funcs)
    if return_astrometry:
        return astrometry
    if return_both:
        return check_ruwe_reduced(*astrometry), astrometry
    return check_ruwe_reduced(*astrometry)

### --- ###
def calculate_ruwe_a0(df=None,period=1e4,a0=1,phot_g_mean_mag=14,Tp=0,ecc=0,omega=0,inc=0,w=0,
                      ra=None,dec=None,parallax=None,pmra=0,pmdec=0,return_astrometry=False,return_both=False):
    if df is not None:
        if phot_g_mean_mag is None:
            phot_g_mean_mag = df["phot_g_mean_mag"]
        if parallax is None:
            parallax = df["parallax"]
        if ra is None:
            ra = df["ra"]
        if dec is None:
            dec = df["dec"]
        ecc = np.minimum(ecc, 0.99)
    astrometry = gaiamock.predict_astrometry_binary_in_terms_of_a0(
                            ra=ra,dec=dec,parallax=parallax,pmra=pmra,pmdec=pmdec,
                            period=period,Tp=Tp,ecc=ecc,omega=omega,inc=inc,w=w, 
                            a0_mas=a0, phot_g_mean_mag=phot_g_mean_mag,data_release='dr3',c_funcs=c_funcs)
    if return_astrometry:
        return astrometry
    if return_both:
        return check_ruwe_reduced(*astrometry), astrometry
    return check_ruwe_reduced(*astrometry)
    
### --- ###
def fit_full_astrometric_cascade(t_ast_yr, psi, plx_factor, ast_obs, ast_err, ruwe_min=1.4, orbital_solution_prange=(1.5,3.5), skip_ruwe=False, skip_full=False):
    '''
        simplified version of fit_full_astrometric_cascade() in Gaiamock, 
        used only to ask what type of solution there is. 
        5: 5 parameter standard solution
        7: 7 parameter acceleration solution
        9: 9 parameter acceleration solution
        12: full orbit solution
    '''    
    N_visibility_periods = int(np.sum( np.diff(t_ast_yr*365.25) > 4) + 1)
    if (N_visibility_periods < 12) or (len(ast_obs) < 13): 
        return 5
    
    if not skip_ruwe:
        # check 5-parameter solution 
        ruwe = check_ruwe_reduced(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err)
        # make RUWE cut
        if ruwe < ruwe_min:
            return 5
    
    # mu is  ra, pmra, pmra_dot, pmra_ddot, dec, pmdec, pmdec_dot, pmdec_ddot, plx
    F2_9par, s_9par, mu, sigma_mu = gaiamock.check_9par(t_ast_yr, psi, plx_factor, ast_obs, ast_err)
    plx_over_err9 = mu[-1]/sigma_mu[-1]
    if (F2_9par < 25) and (s_9par > 12) and (plx_over_err9 > 2.1*s_9par**1.05):
        return 9
    
    # mu is ra, pmra, pmra_dot, dec, pmdec, pmdec_dot, plx
    F2_7par, s_7par, mu, sigma_mu = gaiamock.check_7par(t_ast_yr, psi, plx_factor, ast_obs, ast_err)
    plx_over_err7 = mu[-1]/sigma_mu[-1]
    if (F2_7par < 25) and (s_7par > 12) and (plx_over_err7 > 1.2*s_7par**1.05):
        return 7
    
    if skip_full:
        return 5
    
    res = gaiamock.fit_orbital_solution_nonlinear(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, c_funcs = c_funcs, L = np.array([10, 0, 0]))
    
    # get the linear parameters 
    period, phi_p, ecc = res
    chi2, mu_linear = gaiamock.get_astrometric_chi2(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, P = period, phi_p = phi_p, ecc = ecc, c_funcs=c_funcs)
    ra_off, pmra, dec_off, pmdec, plx, B, G, A, F = mu_linear
    p0 = [ra_off, dec_off, plx, pmra, pmdec, period, ecc, phi_p, A, B, F, G]
    
    # get some uncertainties 
    errors, a0_mas, sigma_a0_mas, inc_deg = gaiamock.get_uncertainties_at_best_fit_binary_solution(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, p0 = p0, c_funcs = c_funcs)
    #sig_ra,sig_dec,sig_parallax,sig_pmra,sig_pmdec,sig_period,sig_ecc,sig_phi_p,sig_A,sig_B, sig_F,sig_G = errors
    sig_parallax, sig_ecc = errors[2], errors[6]
    Nobs, nu, nu_unbinned = len(ast_obs), len(ast_obs) - 12, len(ast_obs)*8 - 12
    chi2_red_binned = chi2/nu
    
    F2 = gaiamock.predict_F2_unbinned_data(chi2_red_binned = chi2_red_binned, n_param = 12, N_points = Nobs, Nbin=8)
    a0_over_err, parallax_over_error = a0_mas/sigma_a0_mas, plx/sig_parallax

    if (F2 < 25) and (a0_over_err > 158/np.sqrt(period)) and (a0_over_err > 5) and (parallax_over_error > 20000/period) and (sig_ecc < 0.079*np.log(period)-0.244):
        return 12
    return 5

### --- ###
def get_acceleration(t_ast_yr, psi, plx_factor, ast_obs, ast_err, solution_type=None):
    '''
        this is a stripped-down version of check_7par() and check_9par() from gaiamock that just returns accel and jerk magnitudes
    '''
    output = []
    Cinv = np.diag(1/ast_err**2)    
    
    if solution_type == "Acceleration7" or solution_type == "all" or solution_type == "key":
        M = np.vstack([np.sin(psi), t_ast_yr*np.sin(psi), 1/2*t_ast_yr**2*np.sin(psi), np.cos(psi), t_ast_yr*np.cos(psi), 1/2*t_ast_yr**2*np.cos(psi), plx_factor]).T 
        mu = np.linalg.solve(M.T @ Cinv @ M, M.T @ Cinv @ ast_obs)  # ra, pmra, pmra_dot, dec, pmdec, pmdec_dot, plx
        acc = (mu[2]**2 + mu[5]**2)**(1/2)
        if solution_type == "all" or solution_type == "key":
            output.append(acc)
        else:
            return acc
    
    M = np.vstack([np.sin(psi), t_ast_yr*np.sin(psi), 1/2*t_ast_yr**2*np.sin(psi),  1/6*t_ast_yr**3*np.sin(psi), np.cos(psi), t_ast_yr*np.cos(psi), 1/2*t_ast_yr**2*np.cos(psi), 1/6*t_ast_yr**3*np.cos(psi), plx_factor]).T 
    mu = np.linalg.solve(M.T @ Cinv @ M, M.T @ Cinv @ ast_obs)  # ra, pmra, pmra_dot, pmra_ddot, dec, pmdec, pmdec_dot, pmdec_ddot, plx
    acc, jerk = (mu[2]**2 + mu[6]**2)**(1/2), (mu[3]**2 + mu[7]**2)**(1/2)
    if solution_type == "key":
        return output[0], jerk
    if solution_type == "Acceleration9":
        return acc
    elif solution_type == "Jerk9":
        return jerk
    elif solution_type == "Both9":
        return acc, jerk
    else:
        return output[0], acc, jerk
        
### --- ###
def determine_solution_type(solve_type, skip_full, df=None, os_prange=(1,4), **kwargs):
    if solve_type == "ss":
        ruwe, astrometry = calculate_ruwe_ss(df=df, return_both=True, **kwargs)
    elif solve_type == "a0":
        ruwe, astrometry = calculate_ruwe_a0(df=df, return_both=True, **kwargs)
    else:
        ruwe, astrometry = calculate_ruwe(df=df, return_both=True, **kwargs)
    
    temp_skip_full = skip_full
    if "period" in kwargs.keys():
        logperiod = np.log10(kwargs["period"])
        if not (os_prange[0] <= logperiod and logperiod <= os_prange[1]):
            temp_skip_full = True
    if ruwe > 1.4:
        return fit_full_astrometric_cascade(*astrometry, skip_ruwe=True, skip_full=temp_skip_full)
    return 5

### --- ###
def determine_accel(accel_type, df=None, **kwargs):
    astrometry = calculate_ruwe(df=df, return_astrometry=True, **kwargs)
    return get_acceleration(*astrometry, solution_type=accel_type)

### --- MARGINALIZATION PIPELINE --- ###
### --- ###
def setup_marginalize(kwargs, marginalize_angles=False, marginalize_pm=False, marginalize_position=False):
    '''
        This function takes the setting keywords and translates them
        into what is passed into further functions
        
        The parameters are:
            marginalize_angles (False): If set to true, uses built-in functions to
                sample w, omega, inclination, and Time of Periastron and marginalize
                over them. This means you don't have to manually set each of them in 
                the keywords of the function call
            marginalize_pm (False): If set to true, samples pmra and pmdec from the
                provided catalogue
            marginalize_position (False): If set to true, samples ra and dec from the
                provided catalogue
        catalogue (string) (defaults to None): path to catalogue if using the 
            marginalize_pm or marginalize_position setting
    '''
        
    # split into marginalized and un-marginalized parameters
    marg_funcs = dict()
    static_params = dict()
    
    # set default angle distributions
    if marginalize_angles:
        marg_funcs["w"] = random_angle
        marg_funcs["omega"] = random_angle
        marg_funcs["inc"] = random_inc
        marg_funcs["Tp"] = random_Tp
    
    # if default pmra,pmdec sampling is chosen
    if marginalize_pm:
        marg_funcs["pmra"] = "catalogue"
        marg_funcs["pmdec"] = "catalogue"
    
    # if default ra,dec sampling is chosen
    if marginalize_position:
        marg_funcs["ra"] = "catalogue"
        marg_funcs["dec"] = "catalogue"

    # set the rest of the parameters
    for kwarg in kwargs.keys():
        outkwarg = kwarg
        # some aliases
        if kwarg == "g":
            outkwarg = "phot_g_mean_mag"
        if kwarg == "p":
            outkwarg = "period"
        if callable(kwargs[kwarg]):
            marg_funcs[outkwarg] = kwargs[kwarg]
        else:
            static_params[outkwarg] = kwargs[kwarg]
    
    return static_params, marg_funcs

### --- ###
def generate_marginalized_values(marg_funcs, df=None, catalogue=None, noise=None, count=1):
    '''
        takes all the marginalization functions and samples
        values from them. If a catalogue is provided, the 
        shortcut "catalogue" allows you to sample from a 
        catalogue
        
        inputs:
            marg_funcs {parameter_name: function()}        
        outputs:
            marg_vals {parameter_name: float}
        
    '''
    marg_vals = []
    for i in range(count):
        marg_vals_dict = dict()
        for arg in marg_funcs.keys():
            if catalogue is not None:
                catdx = np.random.randint(len(catalogue)) # we want the same entry to be taken each time
            if isinstance(marg_funcs[arg], str): 
                temparg = arg
                if arg == "ecc":
                    temparg = "eccentricity"
                if arg == "Tp":
                    temparg = "t_periastron"
                if arg == "inc":
                    temparg = "inclination"
                if arg == "w":
                    temparg = "arg_periastron"  
                if arg == "q":
                    temparg = "mass_ratio"  
                    
                if marg_funcs[arg] == "catalogue":
                    marg_vals_dict[arg] = catalogue[catdx][temparg]
                if marg_funcs[arg] == "df":
                    marg_vals_dict[arg] = df[temparg]
                if arg == "noise":
                    marg_vals_dict[arg] = sample_with_noise(*noise[arg])
            else:
                marg_vals_dict[arg] = marg_funcs[arg]()
        marg_vals.append(marg_vals_dict)
    
    return marg_vals
    
### --- ###
def marginalize(func, sample_count=100, return_count=1, function_count=1, set_parameters=None, noise=None,
                df=None, marginalize_angles=False, marginalize_pm=False, marginalize_position=False,
                catalogue=None, return_params=False, func_params=[], pbar=None, verbose=True,
                **kwargs):
    '''
        Now pay attention because this one is important. This function 
        repeatedly calls another function and marginalizes over some of
        the inputs of that function. Here's how it works
        
        func: the function you want to marginalize over. Make sure it has
            a bunch of keywords with default values
        **kwargs: each keyword you put in here should be a keyword in your
            function. There are two options, you can either put in a float
            as a new default value, or you can put in a function which
            describes how to randomly sample that parameter. You can also pass
            in the string "catalogue", if paired with a catalogue will sample 
            this parameter from that catalogue
        
        other inputs:
            sample_count: amount of times to call the function, default is 1000
            noise: dictionary ({[parameter name]: (location, noise amount %), ...})
                use in conjunction with kwarg [parameter name]="noise"
            df: a row of a table from a catalogue. This is just a shortcut for
                creating a bunch of default values if using my gaiamock calling
                functions
            marginalize_angles, marginalize_pm, marginalize_position: settings
                for using built-in functions for marginalization
            marginalize_angles: sample w,omega,inc,Tp
            marginalize_pm: sample pmra, pmdec
            marginalize_position: sample ra, dec
            catalogue: used if sampling a parameter from a catalogue. Needed
                if using marginalize_pm or marginalize_position
            return_params: returns the values of the marginalized parameters for
                each function call
        
        outputs:
            values (list, length=sample_count)
            (optional, enabled by return_params=True) marginalized_set 
                (list, size=(sample_count,#marginalized_parameters))       
    '''
    
    # set up all the settings and split **kwargs into the parameters
    # that are just new default values, and the ones we marginalize
    static_params, marg_funcs = setup_marginalize(kwargs,
                                            marginalize_angles=marginalize_angles, marginalize_pm=marginalize_pm, marginalize_position=marginalize_position)
    
    # you can inherit a pbar, used if marginalizing over a grid of values in marginalize_grid1d()
    if pbar is None and verbose:
        pbar = tqdm(total=sample_count*function_count)
        
    marginalized_set = generate_marginalized_values(marg_funcs, df=df, catalogue=catalogue, count=sample_count, noise=noise)
    results = []
    for funcidx in range(function_count):
        working_function = func
        if function_count > 1:
            working_function = func[funcidx]
        working_func_params = func_params
        if function_count > 1:
            working_func_params = func_params[funcidx]
        outarr = np.zeros((return_count, sample_count))
        for i in range(sample_count):
            # sample marginalized parameters from provided functions in *args
            if set_parameters is not None:
                marg_vals = set_parameters[i]
            else:
                marg_vals = marginalized_set[i]
            
            outarr[:,i] = working_function(*working_func_params, df=df, **marg_vals, **static_params)
            if verbose:
                pbar.update(1) # move progress bar
        
        #if sample_count == 1:
            #outarr = outarr[:,0]
        if return_count == 1:
            outarr = outarr[0]
            
        outarr = np.array(outarr)
        if return_params:
            results.append((outarr, marginalized_set))
        else:
            results.append(outarr)
    
    if function_count == 1:
        return results[0]
    return results

### --- ###
def _marginalize_worker(i, param_names, param_grids, kwargs, func, df, multiple_df, return_params, seed):
    '''
        this is just a wrapper for marginalize() used in marginalize_grid1d() to parallelise
    '''
    
    np.random.seed(seed) # need to give a random seed, otherwise the same random numbers are generated for each worker
    for j, param_name in enumerate(param_names):
        kwargs[param_name] = param_grids[j][i]

    tempdf = df # if supplying dataframes for each marginalize() call
    if multiple_df:
        tempdf = df[i]

    results = marginalize(
        func, df=tempdf,
        return_params=return_params,
        pbar=None,
        verbose=False,
        **kwargs
    )

    if return_params:
        return i, results[0], results[1]
    else:
        return i, results
    
### --- ###
def marginalize_grid1d(func, *grid_params, return_params=False, verbose=True, pbar=None, df=None, multiple_df=None, **kwargs):

    '''
        call marginalize() for different values of a particular parameter provided in grid_param
        
        inputs:
            grid_param (string, list): name of the parameter and list of values it takes on
                you can pass any number of parameters here, and it will iterate through them
                all in the same sequence
            multiple_df: if set to True, you can pass multiple rows into df and have each
                marginalization run use the corresponding row
            
            The rest of the inputs can be read about in the docstring for marginalize()
    '''
    
    param_names = [param[0] for param in grid_params]
    param_grids = [param[1] for param in grid_params]
    parameter_count = len(param_grids[0])
    
    results_buffer = [None] * parameter_count

    if pbar is None and verbose:
        pbar = tqdm(total=parameter_count)

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                _marginalize_worker, i, param_names, param_grids, kwargs.copy(), func,
                df, multiple_df, return_params, np.random.randint(10000)
            )
            for i in range(parameter_count)
        ]

        for future in futures:
            i, *result_parts = future.result()
            if return_params:
                results_buffer[i] = (result_parts[0], result_parts[1])
            else:
                results_buffer[i] = result_parts[0]
            if verbose:
                pbar.update(1)

    if return_params:
        grid1d, marginalized_param_grid = zip(*results_buffer)
        return np.array(grid1d), np.array(marginalized_param_grid)
    else:
        return np.array(results_buffer)

### --- ###
def marginalize_grid2d(func, grid_param, *grid_params2, sample_count=100, return_params=False, verbose=True, df=None, multiple_df=False, return_count=1, pbar=None, **kwargs):

    '''
        call marginalize() over a grid of two parameters provided in grid_param and grid_param2
        
        inputs:
            grid_param (string, list): name of first parameter and list of values it takes on
            grid_param2 (string, list): name of second parameter and list of values it takes on
                you can provide any number of parameters here to be iterated through together
            multiple_df: if set to True, pass in two dimensional array of df rows to have a particular
                row called for each marginalization call
            
            The rest of the inputs can be read about in the docstring for marginalize()
        
        outputs:
            grid2d [grid_param, grid_param2]
    '''
    
    marginalized_param_grid = []
    
    param_name = grid_param[0]
    param_grid = grid_param[1]

    if return_count > 1:
        grid2d = np.zeros((len(param_grid), len(grid_params2[0][1]), return_count, sample_count))
    else:
        grid2d = np.zeros((len(param_grid), len(grid_params2[0][1]), sample_count))
    
    if verbose and pbar is None:
        pbar = tqdm(total=np.prod(grid2d.shape[:-1])//return_count)
    for i, param_val in enumerate(param_grid):
        marginalized_param_grid.append([])
        kwargs[param_name] = param_val
        
        tempdf = df
        if multiple_df:
            tempdf = df[i]
                
        results = marginalize_grid1d(func, *grid_params2, df=tempdf, sample_count=sample_count, return_params=return_params,
                                        pbar=pbar, verbose=verbose, return_count=return_count, **kwargs)
        
        calculated_values = results
        if return_params:
            calculated_values = results[0]
            marginalized_param_grid[i].append(results[1])
        grid2d[i] = calculated_values
        
    if return_params:
        return np.array(grid2d), np.array(marginalized_param_grid)
    else:
        return np.array(grid2d)
    
def marginalize_grid3d(func, grid_param, grid_param2, *grid_params3, sample_count=100, return_count=1, return_params=False, verbose=True, df=None, multiple_df=False, **kwargs):
    marginalized_param_grid = []
    
    param_name = grid_param[0]
    param_grid = grid_param[1]

    if return_count > 1:
        grid3d = np.zeros((len(param_grid), len(grid_param2[1]), len(grid_params3[0][1]), return_count, sample_count))
    else:
        grid3d = np.zeros((len(param_grid), len(grid_param2[1]), len(grid_params3[0][1]), sample_count))
    
    pbar = None
    if verbose:
        pbar = tqdm(total=np.prod(grid3d.shape[:-1])/return_count)
    for i, param_val in enumerate(param_grid):
        marginalized_param_grid.append([])
        kwargs[param_name] = param_val
        
        tempdf = df
        if multiple_df:
            tempdf = df[i]
                
        results = marginalize_grid2d(func, grid_param2, *grid_params3, df=tempdf, sample_count=sample_count, return_count=return_count, return_params=return_params,
                                        pbar=pbar, verbose=verbose, **kwargs)
        
        calculated_values = results
        
        if return_params:
            calculated_values = results[0]
            marginalized_param_grid[i].append(results[1])
        grid3d[i] = calculated_values
        
    if return_params:
        return np.array(grid3d), np.array(marginalized_param_grid)
    else:
        return np.array(grid3d)
### ----------------- ###
###
### Wrapper functions
###
### ----------------- ###

### --- RUWE --- ###
''' 
    single_star (boolean): defaults to False
    This tells the program if you want to use
    Gaiamock on single stars or binaries
    by switching between calculate_ruwe()
    and calculate_ruwe_ss()
    
    use_a0 (boolean): defaults to False
    This tells the program if instead of using
    M1,q you want to use a0 for calling gaiamock.
    Just switches to using calcualte_ruwe_a0()
    if set to True
'''
### --- ###
def ruwe_type_func(ruwe_type):
    func = calculate_ruwe
    if ruwe_type == "ss":
        func = calculate_ruwe_ss
    if ruwe_type == "a0":
        func = calculate_ruwe_a0
    return func

### --- ###
def marginalize_ruwe(ruwe_type="normal", **kwargs):
    return marginalize(ruwe_type_func(ruwe_type), **kwargs)

### --- ###
def marginalize_ruwe_grid1d(*grid_params, ruwe_type="normal", **kwargs):
    return marginalize_grid1d(ruwe_type_func(ruwe_type), *grid_params, **kwargs)

### --- ###
def marginalize_ruwe_grid2d(grid_param, *grid_params2, ruwe_type="normal", **kwargs):  
    return marginalize_grid2d(ruwe_type_func(ruwe_type), grid_param, *grid_params2, **kwargs)

### --- ASTROMETRIC SOLUTIONS --- ###
''' These simply marginalize using determine_solution_type()
'''

### --- ###
def bin_solutions(arr):
    hist, _ = np.histogram(arr, bins=[5,7,9,12,13])
    return hist

### --- ###
def convert_to_type_arr(marginalized_arr):
    '''
        this function converts an n-dimensional array with a finite possible set of values
        into an array containing the counts for each value
    '''
    arrshape = len(marginalized_arr.shape)
    if arrshape == 1:
        return bin_solutions(marginalized_arr)
    elif arrshape == 2:
        new_solution_chances = np.zeros((*marginalized_arr.shape[:-1],4))
        for i in range(marginalized_arr.shape[0]):
            new_solution_chances[i] = bin_solutions(marginalized_arr[i])/marginalized_arr.shape[-1]
        return marginalized_arr  
    elif arrshape == 3:
        new_solution_chances = np.zeros((*marginalized_arr.shape[:-1],4))
        for i in range(marginalized_arr.shape[0]):
            for j in range(marginalized_arr.shape[1]):
                new_solution_chances[i,j] = bin_solutions(marginalized_arr[i,j])/marginalized_arr.shape[-1]
        return marginalized_arr    

### --- ###
def marginalize_solution_type(solve_type="normal", skip_full=False, **kwargs):
    return marginalize(determine_solution_type, func_params=[solve_type, skip_full], **kwargs)

### --- ###
def marginalize_solution_type_grid1d(*grid_params, solve_type="normal", skip_full=False, **kwargs):
    return marginalize_grid1d(determine_solution_type, *grid_params, func_params=[solve_type, skip_full], **kwargs)

### --- ###
def marginalize_solution_type_grid2d(grid_param, *grid_params2, solve_type="normal", skip_full=False, **kwargs):
    return marginalize_grid2d(determine_solution_type, grid_param, *grid_params2, func_params=[solve_type, skip_full], **kwargs)

### --- ###
def marginalize_solution_type_grid3d(grid_param, grid_param2, *grid_params3, solve_type="normal", skip_full=False, **kwargs):
    return marginalize_grid3d(determine_solution_type, grid_param, grid_param2, *grid_params3, func_params=[solve_type, skip_full], **kwargs)

### --------------------- ###
### --- CUBE CREATION --- ###
### --------------------- ###

### --- ###
def convert_to_probability(multiarr, bincount=100, trim=(0,95)):
    ''' Converts a multi dimensional array of calculated values into a multi dimensional array with probability densities
    Notably, probability densities will need to be re-multiplied by bin size to reobtain normalization
    
    Also of note, this is a general function but it's intended for the log10 of a RUWE distribution
    '''

    set_bins = np.linspace(np.percentile(multiarr, trim[0]), np.percentile(multiarr, trim[1]), bincount) #cutoff top 100-trim% of values to not get those crazy outliers
    bin_sizes = 10**set_bins[1:] - 10**set_bins[:-1] # in linear bin size
    
    probability_marr = np.zeros((*multiarr.shape[:-1], bincount-1))
    for i in range(multiarr.shape[0]):
        for j in range(multiarr.shape[1]):
            for k in range(multiarr.shape[2]):
                data = multiarr[i,j,k,]
                histograms = np.histogram(data, bins=set_bins)[0] # create binned distribution
                probabilities = histograms/len(data) # divide by total number of calculations to get a probability
                probability_marr[i][j][k] = probabilities / bin_sizes # scale by linear bin size
    
    return probability_marr, set_bins

### --- ###
def create_ruwe_cube(grid_param, grid_param2, *grid_params3, sample_count=100, bincount=100, trim=(0,95), return_bins=False, gaussian=None, gaussian_calls=10, **kwargs):
    '''
        calls marginalize_ruwe_grid2d(), works exactly the same as that function
        but then also converts the RUWE samples to a probability distribution
        
        reccomended to set return_bins=True, since the bins will be important for plotting
    '''
    ruwes = marginalize_grid3d(calculate_ruwe, grid_param, grid_param2, *grid_params3, sample_count=sample_count, **kwargs)
    
    if gaussian is not None:
        tempruwes = np.zeros((*ruwes.shape[:3], ruwes.shape[3]*gaussian_calls))
        for i in range(ruwes.shape[0]):
            for j in range(ruwes.shape[1]):
                for k in range(ruwes.shape[2]):
                    for l in range(ruwes.shape[3]):
                        for m in range(gaussian_calls):
                            tempruwes[i,j,k,l*gaussian_calls+m] = ruwes[i,j,k,l] + np.random.normal(loc=gaussian[0], scale=gaussian[1])
        ruwes = tempruwes
        
    probabilities, set_bins = convert_to_probability(np.log10(ruwes), bincount=bincount, trim=trim)
    
    if return_bins:
        return probabilities, set_bins
    else:
        return probabilities
    
### --- ###
def generate_mock_data(sample_count, catalogue_with_masses, binary_fraction=0.5, p_model=None, s_model=(0.15,0.1), eccentricity=None, verbose=True, noise=True, only_ruwe=False, save_colour=False):
    if verbose:
        print("precomputing all orbital parameters...")
    outlog = []
    # precompute all random decisions except for angles
    binary_count = int(binary_fraction*sample_count)
    is_binary = np.array([True for _ in range(binary_count)] + [False for _ in range(sample_count-binary_count)])
    np.random.shuffle(is_binary)
    catalogue_entries = np.random.choice(catalogue_with_masses, sample_count)
    
    # eccentricity
    eccs = np.zeros(binary_count)
    if eccentricity is not None:
        eccs = sample_perfect(eccentricity, (0,0.95), binary_count, 1)
    np.random.shuffle(eccs)
    
    logperiods = np.zeros(binary_count)
    if p_model is not None:
        logperiods = sample_perfect(p_model[0], (1,8), binary_count, *p_model[1])
    np.random.shuffle(logperiods)
    # logperiod = np.random.normal(loc=p_model[0], scale=p_model[1]) # days
    #     while logperiod > 8 or logperiod < 1:
    #         logperiod = np.random.normal(loc=p_model[0], scale=p_model[1])
    #     period = 10**logperiod
            
    binary_counter = 0    
    
    if verbose:
        print("calculating astrometric signals...")
    for i in tqdm(range(sample_count)):
        outry = dict()
        catalogue_entry = catalogue_entries[i]
        outry["ra"], outry["dec"] = float(catalogue_entry["ra"]), float(catalogue_entry["dec"])
        outry["parallax"] = float(catalogue_entry["parallax"])
        outry["iso_mass"] = float(catalogue_entry["iso_masses"])
        outry["phot_g_mean_mag"] = float(catalogue_entry["phot_g_mean_mag"])
        outry["is_binary"] = bool(is_binary[i])
        outry["solution_type"] = 5
        if save_colour:
            outry["bp_rp"] = float(catalogue_entry["bp_rp"])
            
        ruwe = 0
        if noise:
            # unmodelled noise
            ruwe = np.random.normal(loc=s_model[0], scale=s_model[1])
        
        # now add ruwe contribution from the object itself
        if is_binary[i]:
            ecc = eccs[binary_counter]
            period = 10**logperiods[binary_counter]
            
            q = np.random.rand() # random between 0 and 1
            # q = np.random.normal(loc=0.2, scale=0.07) # days
            # while q > 1 or q < 0:
            #     q = np.random.normal(loc=0.2, scale=0.07)
            
            #while q*outry["iso_mass"] > 0.08: #bigger than a brown dwarf?
            inc = float(random_inc())
            omega, Omega = float(random_angle()), float(random_angle())
            Tp = float(random_Tp()*period)
            
            if only_ruwe:
                calc_ruwe = marginalize_ruwe(
                                    sample_count=1, 
                                    m1=outry["iso_mass"], period=period, q=q, ecc=ecc, inc=inc, w=omega, omega=Omega, Tp=Tp, f=1e-2,
                                    ra=catalogue_entry["ra"], dec=catalogue_entry["dec"], parallax=catalogue_entry["parallax"],
                                    phot_g_mean_mag=catalogue_entry["phot_g_mean_mag"], verbose=False)
            else:
                # use gaiamock to get every astrometric signature. We might as well compute all the signatures for all the objects
                calc_ruwe, solution_type, outry["acceleration7"], outry["acceleration9"], outry["jerk"] = [float(pram) for pram in marginalize(
                                        [calculate_ruwe, determine_solution_type, determine_accel, determine_accel, determine_accel],
                                        function_count=5, func_params=[[],["normal", False], ["Acceleration7"], ["Acceleration9"], ["Jerk9"]], sample_count=1, 
                                        m1=outry["iso_mass"], period=period, q=q, ecc=ecc, inc=inc, w=omega, omega=Omega, Tp=Tp, f=1e-2,
                                        ra=catalogue_entry["ra"], dec=catalogue_entry["dec"], parallax=catalogue_entry["parallax"],
                                        phot_g_mean_mag=catalogue_entry["phot_g_mean_mag"], verbose=False)]
                outry["solution_type"] = int(solution_type)
            ruwe += calc_ruwe
            
            # generally we wouldn't have these if there's no orbit solution
            # but it's useful to have for debugging
            outry["period"] = float(period)
            outry["eccentricity"] = float(ecc)
            outry["mass_ratio"] = float(q)
            outry["inclination"] = inc
            outry["omega"] = omega
            outry["Omega"] = Omega
            outry["t_periastron"] = Tp
            
            binary_counter += 1
            
        else: # single star, make it just follow the gaussian
            ruwe += calculate_ruwe_ss(df=outry)
            
        outry["ruwe"] = ruwe
        outlog.append(outry)
    return np.array(outlog)