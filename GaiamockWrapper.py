import numpy as np
import gaiamock.gaiamock as gaiamock
import pickle
import numbers
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    catalogue through the keyword "catalogue_path"
'''
### --- ###
def random_angle():
    return np.random.rand()*2*np.pi

def random_inc():
    return np.random.rand()*0.5*np.pi

def random_Tp():
    return np.random.rand()

### --- BUILT-IN FUNCTIONS FOR MARGINALIZING --- ###
### --- ###
def calculate_ruwe(df=None,m1=0.4,q=0.2,period=1e4,ecc=0,inc=0,Tp=0,omega=0,w=0,f=1e-2,phot_g_mean_mag=14,parallax=None,ra=None,dec=None,pmra=0,pmdec=0,return_astrometry=False):
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
    astrometry = gaiamock.predict_astrometry_luminous_binary(ra = ra, dec = dec, parallax = parallax, 
                        pmra = pmra, pmdec = pmdec, m1 = m1, m2 = q*m1, period = period, Tp = Tp*period, ecc = ecc, 
                        omega = omega, inc = inc, w = w, phot_g_mean_mag = phot_g_mean_mag, f = f, data_release = 'dr3',
                        c_funcs = c_funcs)
    if return_astrometry:
        return astrometry
    return gaiamock.check_ruwe(*astrometry)[0]

### --- ###
def calculate_ruwe_ss(df=None,phot_g_mean_mag=None,parallax=None,ra=None,dec=None,pmra=0,pmdec=0):
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
    return gaiamock.check_ruwe(*astrometry)[0]

### --- ###
def calculate_ruwe_a0(df=None,period=1e4,a0=1,phot_g_mean_mag=14,Tp=0,ecc=0,omega=0,inc=0,w=0,
                      ra=None,dec=None,parallax=None,pmra=0,pmdec=0,return_astrometry=False):
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
    return gaiamock.check_ruwe(*astrometry)[0]
    
### --- ###
def fit_full_astrometric_cascade(t_ast_yr, psi, plx_factor, ast_obs, ast_err, binned = True, ruwe_min = 1.4):
    '''
        simplified version of fit_full_astrometric_cascade() in Gaiamock, 
        used only to ask if there is a solution. If no solution, return 0
        if there is a solution (7 or 9 parameter), return 1
    '''    
    N_visibility_periods = int(np.sum( np.diff(t_ast_yr*365.25) > 4) + 1)
    if (N_visibility_periods < 12) or (len(ast_obs) < 13): 
        return 0
    
    # check 5-parameter solution 
    ruwe, mu, sigma_mu = gaiamock.check_ruwe(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, binned=binned)
    # make RUWE cut
    if ruwe < ruwe_min:
        return 0
    
    # mu is  ra, pmra, pmra_dot, pmra_ddot, dec, pmdec, pmdec_dot, pmdec_ddot, plx
    F2_9par, s_9par, mu, sigma_mu = gaiamock.check_9par(t_ast_yr, psi, plx_factor, ast_obs, ast_err, binned=binned)
    plx_over_err9 = mu[-1]/sigma_mu[-1]
    if (F2_9par < 25) and (s_9par > 12) and (plx_over_err9 > 2.1*s_9par**1.05):
        return 1
    
    # mu is ra, pmra, pmra_dot, dec, pmdec, pmdec_dot, plx
    F2_7par, s_7par, mu, sigma_mu = gaiamock.check_7par(t_ast_yr, psi, plx_factor, ast_obs, ast_err, binned=binned)
    plx_over_err7 = mu[-1]/sigma_mu[-1]
    if (F2_7par < 25) and (s_7par > 12) and (plx_over_err7 > 1.2*s_7par**1.05):
        return 1

    return 0

### --- ###
def determine_if_astrometric_solution(df, **kwargs):
    astrometry = calculate_ruwe(df=df, return_astrometry=True, **kwargs)
    return fit_full_astrometric_cascade(*astrometry)

### --- ###
def determine_if_astrometric_solution_a0(df, **kwargs):
    astrometry = calculate_ruwe_a0(df=df, return_astrometry=True, **kwargs)
    return fit_full_astrometric_cascade(*astrometry)

### --- MARGINALIZATION PIPELINE --- ###
### --- ###
def setup_marginalize(kwargs, marginalize_angles=False, marginalize_pm=False, marginalize_position=False,
                     catalogue_path=None):
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
        catalogue_path (string) (defaults to None): path to catalogue if using the 
            marginalize_pm or marginalize_position setting
    '''
    # pull out catalogue (if you need it)
    catalogue=None
    if catalogue_path is not None:
        infile = open(catalogue_path, "rb")
        catalogue = pickle.load(infile)
        infile.close()
        
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
        if isinstance(kwargs[kwarg], numbers.Number):
            static_params[outkwarg] = kwargs[kwarg]
        else:
            marg_funcs[outkwarg] = kwargs[kwarg]
    
    return static_params, marg_funcs, catalogue

### --- ###
def generate_marginalized_values(marg_funcs, df=None, catalogue=None, count=1):
    '''
        takes all the marginalization functions and samples
        values from them. It a catalogue is provided, the 
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
            else:
                marg_vals_dict[arg] = marg_funcs[arg]()
        marg_vals.append(marg_vals_dict)
    
    if count == 1:
        return marg_vals[0]
    else:
        return marg_vals

### --- ###
def marginalize(func, sample_count=100, set_parameters=None,
                df=None, marginalize_angles=False, marginalize_pm=False, marginalize_position=False,
                catalogue_path=None, return_params=False, pbar=None, verbose=True,
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
            df: a row of a table from a catalogue. This is just a shortcut for
                creating a bunch of default values if using my gaiamock calling
                functions
            marginalize_angles, marginalize_pm, marginalize_position: settings
                for using built-in functions for marginalization
            marginalize_angles: sample w,omega,inc,Tp
            marginalize_pm: sample pmra, pmdec
            marginalize_position: sample ra, dec
            catalogue_path: used if sampling a parameter from a catalogue. Needed
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
    static_params, marg_funcs, catalogue = setup_marginalize(kwargs,
                                            marginalize_angles=marginalize_angles, marginalize_pm=marginalize_pm, marginalize_position=marginalize_position,
                                            catalogue_path=catalogue_path)
    
    # you can inherit a pbar, used if marginalizing over a grid of values in marginalize_grid1d()
    if pbar is None and verbose:
        pbar = tqdm(total=sample_count)
        
    marginalized_set = generate_marginalized_values(marg_funcs, df=df, catalogue=catalogue, count=sample_count)
    outarr = np.zeros(sample_count)
    for i in range(sample_count):
        # sample marginalized parameters from provided functions in *args
        if set_parameters is not None:
            marg_vals = set_parameters[i]
        else:
            marg_vals = marginalized_set[i]
        
        outarr[i] = func(df=df, **marg_vals, **static_params)
        if verbose:
            pbar.update(1) # move progress bar
    
    if return_params:
        return outarr, marginalized_set
    else:
        return outarr
    
### --- ###
def _marginalize_worker(i, param_names, param_grids, kwargs, func, sample_count, df,
                        multiple_df, marginalize_angles, marginalize_pm, marginalize_position,
                        catalogue_path, return_params, seed):
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
        func, sample_count=sample_count, df=tempdf,
        marginalize_angles=marginalize_angles,
        marginalize_pm=marginalize_pm,
        marginalize_position=marginalize_position,
        catalogue_path=catalogue_path,
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
def marginalize_grid1d(func, *grid_params, sample_count=100,
                       df=None, marginalize_angles=False, marginalize_pm=False,
                       marginalize_position=False, catalogue_path=None, multiple_df=False,
                       return_params=False, verbose=True, pbar=None, **kwargs):

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
                sample_count, df, multiple_df, marginalize_angles,
                marginalize_pm, marginalize_position, catalogue_path,
                return_params, np.random.randint(10000)
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
def marginalize_grid2d(func, grid_param, *grid_params2, sample_count=100, 
                            df=None, marginalize_angles=False, marginalize_pm=False, marginalize_position=False,
                            catalogue_path=None, multiple_df=False, return_params=False, verbose=True,
                            **kwargs):

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

    grid2d = np.zeros((len(param_grid), len(grid_params2[0][1]), sample_count))
    
    pbar = None
    if verbose:
        pbar = tqdm(total=np.prod(grid2d.shape[:-1]))
    for i, param_val in enumerate(param_grid):
        marginalized_param_grid.append([])
        kwargs[param_name] = param_val
        
        tempdf = df
        if multiple_df:
            tempdf = df[i]
                
        results = marginalize_grid1d(func, *grid_params2, sample_count=sample_count, 
                                        df=tempdf, marginalize_angles=marginalize_angles, marginalize_pm=marginalize_pm, marginalize_position=marginalize_position,
                                        catalogue_path=catalogue_path, return_params=return_params,
                                        pbar=pbar, verbose = verbose,
                                        **kwargs)
        calculated_values = results
        if return_params:
            calculated_values = results[0]
            marginalized_param_grid[i].append(results[1])
        grid2d[i] = calculated_values
                
    if return_params:
        return np.array(grid2d), np.array(marginalized_param_grid)
    else:
        return np.array(grid2d)

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
def marginalize_ruwe(single_star=False, use_a0=False, **kwargs):
    func = calculate_ruwe
    if single_star:
        func = calculate_ruwe_ss
    if use_a0:
        func = calculate_ruwe_a0
        
    return marginalize(func, **kwargs)

### --- ###
def marginalize_ruwe_grid1d(*grid_params, single_star=False, use_a0=False, **kwargs):
    func = calculate_ruwe
    if single_star:
        func = calculate_ruwe_ss
    if use_a0:
        func = calculate_ruwe_a0
    
    return marginalize_grid1d(func, *grid_params, **kwargs)

### --- ###
def marginalize_ruwe_grid2d(grid_param, *grid_params2, single_star=False, use_a0=False, **kwargs):
    func = calculate_ruwe
    if single_star:
        func = calculate_ruwe_ss
    if use_a0:
        func = calculate_ruwe_a0
    
    return marginalize_grid2d(func, grid_param, *grid_params2, **kwargs)

### --- ASTROMETRIC SOLUTIONS --- ###
''' These simply marginalize using determine_if_astrometric_solution()
'''
### --- ###
def search_for_solutions(use_a0=False, **kwargs):
    func = determine_if_astrometric_solution
    if use_a0:
        func = determine_if_astrometric_solution_a0
    return marginalize(func, **kwargs)

### --- ###
def search_for_solutions_grid1d(*grid_params, use_a0=False, **kwargs):
    func = determine_if_astrometric_solution
    if use_a0:
        func = determine_if_astrometric_solution_a0
    return marginalize_grid1d(func, *grid_params, **kwargs)

### --- ###
def search_for_solutions_grid2d(grid_param, *grid_params2, use_a0=False, **kwargs):
    func = determine_if_astrometric_solution
    if use_a0:
        func = determine_if_astrometric_solution_a0
    return marginalize_grid2d(func, grid_param, *grid_params2, **kwargs)

### --------------------- ###
### --- CUBE CREATION --- ###
### --------------------- ###

### --- ###
def convert_to_probability(multiarr, bincount=100, trim=95):
    ''' Converts a multi dimensional array of calculated values into a multi dimensional array with probability densities
    Notably, probability densities will need to be re-multiplied by bin size to reobtain normalization
    
    Also of note, this is a general function but it's intended for the log10 of a RUWE distribution
    '''

    set_bins = np.linspace(np.min(multiarr), np.percentile(multiarr, trim), bincount) #cutoff top 100-trim% of values to not get those crazy outliers
    bin_sizes = 10**set_bins[1:] - 10**set_bins[:-1] # in linear bin size
    
    probability_marr = np.zeros((multiarr.shape[0], multiarr.shape[1], bincount-1))
    for i in range(multiarr.shape[0]):
        for j in range(multiarr.shape[1]):
            data = multiarr[i,j,]
            histograms = np.histogram(data, bins=set_bins)[0]
            probabilities = histograms/len(data)
            probability_marr[i][j] = probabilities / bin_sizes
    
    return probability_marr, set_bins

### --- ###
def create_cube(grid_param, *grid_params2, sample_count=100, bincount=None, trim=95, return_bins=False, **kwargs):
    '''
        calls marginalize_ruwe_grid2d(), works exactly the same as that function
        but then also converts the RUWE samples to a probability distribution
        
        reccomended to set return_bins=True, since the bins will be important for plotting
    '''
    ruwes = marginalize_ruwe_grid2d(grid_param, *grid_params2, sample_count=sample_count, **kwargs)
    
    if bincount is None:
        bincount = sample_count // 5
    probabilities, set_bins = convert_to_probability(np.log10(ruwes), bincount=bincount, trim=trim)
    
    if return_bins:
        return probabilities, set_bins
    else:
        return probabilities