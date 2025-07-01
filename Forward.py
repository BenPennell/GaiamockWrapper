import numpy as np
import pickle
import emcee
from multiprocessing import Pool
from scipy.interpolate import RegularGridInterpolator
import sympy
import GaiamockWrapper as gw
import corner
import matplotlib.pyplot as plt
from matplotlib import colors
import datetime

def calculate_m2_halb(m1, a0, period, parallax):
    # Section 5.3 of Halbwachs 2023 2206.05726. We get a cubic function to solve
    fm = float((a0**3 * 365.25**2) / (period**2 * parallax**3))
    m = sympy.symbols("m", real=True) # guarantee real solutions
    roots = sympy.solve(m**3 - fm*(m**2) - fm*(m1**2) - 2*fm*m*m1)
    return np.max(roots) # take positive root

def thiele_innes(df):
    return df[['a_thiele_innes','b_thiele_innes','c_thiele_innes','f_thiele_innes','g_thiele_innes','h_thiele_innes']]

def calculate_inc(df):
    A,B,C,F,G,H = thiele_innes(df)
    return (A*G - B*F) / ( (A**2 + B**2) * (F**2 + G**2) )

def a0_from_thiele_innes(df):
    A,B,C,F,G,H = thiele_innes(df)
    u = 0.5 * (A**2 + B**2 + F**2 + G**2)
    v = A * G - B * F
    a0_mas = np.sqrt(u + np.sqrt(u**2 - v**2))
    return a0_mas

def get_q(r, binary):
    a = float(a0_from_thiele_innes(binary))
    m1 = float(r[r["SOURCE_ID"] == binary["source_id"]]["iso_masses"][0])
    period = float(binary["period"])
    parallax = float(binary["parallax"])
    m2 = calculate_m2_halb(m1,a,period,parallax)
    return m2/m1

### --- ###
def match_catalogues(r1, r2, column1, column2=None):
    index1 = column1
    index2 = column1
    if column2 is not None:
        index2 = column2
    return r1[r1[index1] == r2[index2]]

### --- ###
def convert_to_usable_catalogue(cat, nss_cat=None, accel_cat=None):
    out_log = dict()
    out_log["parallax"] = cat["parallax"]
    out_log["ra"] = cat["ra"]
    out_log["dec"] = cat["dec"]
    out_log["ruwe"] = cat["ruwe"]
    try:
        out_log["source_id"] = cat["source_id"]
    except:
        out_log["source_id"] = cat["SOURCE_ID"]
    try:
        out_log["iso_mass"] = cat["iso_mass"]
    except: 
        out_log["iso_mass"] = cat["iso_masses"]
    out_log["solution_type"] = 5
    
    if nss_cat is not None:
        out_log["solution_type"] = 12
        out_log["period"] = nss_cat["period"]
        out_log["eccentricity"] = nss_cat["eccentricity"]
        out_log["mass_ratio"] = get_q(cat, nss_cat)
    
    if accel_cat is not None:
        if accel_cat["nss_solution_type"] == "Acceleration7":
            out_log["solution_type"] = 7
            out_log["acceleration"] = np.sqrt(accel_cat["accel_ra"]**2 + accel_cat["accel_dec"]**2)
        elif accel_cat["nss_solution_type"] == "Acceleration9":
            out_log["solution_type"] = 9
            out_log["acceleration9"] = np.sqrt(accel_cat["accel_ra"]**2 + accel_cat["accel_dec"]**2)
            out_log["jerk"] = np.sqrt(accel_cat["deriv_accel_ra"]**2 + accel_cat["deriv_accel_dec"]**2)
    
    for olkey in out_log.keys():
        out_log[olkey] = float(out_log[olkey])
    return out_log
        
### --- ###
def calculate_orbit_parameter(m, q, w):
    ''' This is lambda
    '''
    return q*w*m**(1/3)*(1 + q)**(-2/3)

### --- ###
def w_from_l(l, m, q):
    return l * (1 + q)**(2/3) * m**(-1/3) / q

### --- ###
def q_from_l(l, m, w):
    z = (l/(w*m**(1/3)))**(-3)
    q = sympy.symbols("q", real=True)
    roots = sympy.solve(-z*q**3 - q**2 + 2*q + 1)
    if len(roots) == 0:
        return -1
    return roots[0]

### --- ###
def gaussian(x, mu, sigma):
    return np.exp(-(mu - x)**2/(2*sigma**2)) / np.sqrt(2 * np.pi * sigma**2)

### --- ###
def within_range(value, target_range):
    ''' Just checks if value is within a tuple range.
        Called in within_prior().
        Easier to read if this is a separate function
    '''
    return value >= target_range[0] and value <= target_range[1]

### --- ###
def binned_probability(distribution, target, bin_width):
    '''
        compute the probability density for getting a particular value
        given a distrbution of values
    '''
    # first determine how many times you're in the target bucket
    count = len(distribution[((1-bin_width)*target <= distribution) & (distribution <= (1+bin_width)*target)])
    # divide by size of the bucket and total sample count to get probability density
    return count / (2*bin_width*target) / len(distribution)        

### --- ###
def calculate_q_distribution(qs):
    ''' Edit this function for new models
    
        This function creates a flat q distribution
    '''
    return np.ones(len(qs)) / len(qs)

### --- ###
def e_func(e, e_param):
    return (2*e)**e_param

### --- ###
def calculate_e_distribution(es, e_param):
    probabilities =  e_func(es, e_param)
    return probabilities / np.sum(probabilities)

### --- ###
def calculate_p_distribution(ps, mu, sigma):
    ''' Edit this function for new models
    
        This function creates a log-normal P
        distribution based on the parameters
        given in p_params
        
        And we need to normalize it because we
        cut off the tails of the distribution
    '''
    
    probabilities = gaussian(ps, mu, sigma) 
    return probabilities / np.sum(probabilities) # normalize the function

    
### --- ###
def calculate_model_cube(ps, qs, es, p_params, e_params):
    ''' This function 'is' the model
        p(P, q | model)
        
        What we're going to do is create arrays of our P and q
        distributions. But, we want a 'slice' with every combination
        of p(P)*p(q). So, we will turn one of them into a column vector
        and matrix multiply to get the slice
        
        You will be forced into having the same P resolution, but the q resolution
        can (and perhaps should be) larger than the L resolution in the RUWE cube
    '''
    model_cube = dict() # store all relevant info in a larger structure

    ## q distribution
    qarr = calculate_q_distribution(qs)
    
    ## p distribution (log normal)
    parr = calculate_p_distribution(ps, *p_params)
    
    ## e distribution
    earr = calculate_e_distribution(es, e_params)
    
    qarr = [ [q] for q in qarr ] # turn into a "column"
    slice_data = qarr * earr # [q,e]
    cube_data = np.array([p*slice_data for p in parr]) # [p,q,e]

    model_cube["qs"] = qs
    model_cube["ps"] = ps
    model_cube["es"] = es
    model_cube["data"] = cube_data
    return model_cube # [p,q,e]

### ----------------------------------- ###
### --- LIKELIHOOD HELPER FUNCTIONS --- ###

### --- ###
def convert_to_probability_cube(target_object, ruwe_cube, model_cube):
    ''' We need to convert our model slice p(P, q | model) into the same size,
        shape, and units as the RUWE cubes p(P, L | RUWE). Recall the aim, we want
        to matrix multiply these two arrays, so they have to be the same. 
        
        This is the scheme:
        We will take each entry in the (P, q) cube and convert q into L. We will
        then find the appropriate L space to throw the associated value into.
        After this, there are two considerations. If we double up in one image bucket,
        we will sum over all values put into it. If we miss a bucket, we will just
        set its probability to zero. 
        
        It seems that this will be the most expensive step, so it would be nice
        to be able to do this calculation as efficiently as possible
    '''
    
    qs = model_cube["qs"]
    mass = target_object["iso_mass"]
    ls = [calculate_orbit_parameter(mass, q, target_object["parallax"]) for q in qs]
    
    target_ls = ruwe_cube["lambdas"]
    
    probability_cube = np.zeros(ruwe_cube["data"].shape[0:3])  # Shape is (P, L)
    
    # Iterate over each L value and find the corresponding columns
    for col_idx, l in enumerate(ls):
        # Find closest target L
        closest_l_idx = np.argmin(np.abs(l - target_ls))
        
        # Sum the corresponding column values in model_slice into probability_slice
        probability_cube[:,closest_l_idx,:] += model_cube["data"][:,col_idx,:]
    
    return probability_cube # (P, L)

### --- ###
def get_slice(hcube, target_object, parameter):
    ''' uses the RUWE for a given object to 
        get the right slice of the nss cube
    '''
    bins = hcube["bins"]     
    nss_slice = hcube["data"][:,:,:,np.minimum(len(bins[bins < np.log10(target_object[parameter])])-1, hcube["data"].shape[-1]-1)]
    return nss_slice

### --- ###
def interpolate_grid(arr, axs, val):
    interpolator = RegularGridInterpolator(axs, arr, bounds_error=False, fill_value=None)
    return interpolator([val])[0]

### --- ###
def l_known_params(target_object, es, nss_cube, *cubes):    
    lbda = calculate_orbit_parameter(target_object["iso_mass"], target_object["mass_ratio"], target_object["parallax"])
    prd = target_object["period"]
    
    e_index = np.argmin(abs(target_object["eccentricity"] - es))
    
    # pull values from model, cube, and sc cube
    prod = 1
    for working_cube in cubes:
        working_slice = working_cube[:,:,e_index]
        term = interpolate_grid(working_slice, (nss_cube["periods"], nss_cube["lambdas"]), [prd, lbda])
        prod = prod*term
    return prod
    

### --- LIKELIHOOD HELPER FUNCTIONS --- ###
### ----------------------------------- ###

### --- ###
def catalogue_object_likelihood(target_object, model_cube, nss_hcube, sc_hcube, a_hcube, j_hcube, f, ss_params, es, solution_types=[5,7,9,12], ignore_orbital_solutions=False):
    ''' calculate the probability of a particular ruwe detection
        this function evaluates \int p(R | P, M, q, w) p(P, q | model)
        by simply multiplying the two arrays together and summing them
        
        we take a ruwe cube as input, and we need to take the corresponding slice
        and simply multiply it by the weighting array generated by the model
        
        This is for the non single star case. We also need to incorporate the single
        star case, where p(R) is predetermined from the known RUWE distribution of objects.
        We then add them together, taking into consideration the binary fraction
    '''    
    # get the probability and nss slices to be ready to create likelihood terms
    nss_slice = get_slice(nss_hcube, target_object, "ruwe")
    probability_slice = convert_to_probability_cube(target_object, nss_hcube, model_cube)
    solution_index = solution_types.index(target_object["solution_type"])
    sc_slice = sc_hcube["data"][:,:,:,solution_index]
    
    # return standard likelihood if we don't care about orbital solutions
    # or if RUWE is too low (<1.4) to qualify for an orbit solution
    if ignore_orbital_solutions or target_object["ruwe"] < 1.4:
        lss = gaussian(target_object["ruwe"], *ss_params)
        lnss = np.sum(nss_slice*probability_slice)  
        return (1-f)*lss + f*lnss
    
    # now check for particular solution types, and product their corresponding astrometric signals
    elif target_object["solution_type"] == 5:
        lss = gaussian(target_object["ruwe"], *ss_params)
        lnss = np.sum(nss_slice*sc_slice*probability_slice)
        return (1-f)*lss + f*lnss
    elif target_object["solution_type"] == 7:
        a_slice = get_slice(a_hcube, target_object, "acceleration7")
        return f*np.sum(nss_slice*sc_slice*probability_slice*a_slice)
    elif target_object["solution_type"] == 9:
        j_slice = get_slice(j_hcube, target_object, "jerk")
        #a9_slice = get_slice(a9_cube, target_object, "acceleration9")
        return f*np.sum(nss_slice*sc_slice*probability_slice*j_slice)#*a9_slice)
    elif target_object["solution_type"] == 12:
        return f*l_known_params(target_object, es, nss_hcube, sc_slice, nss_slice, probability_slice)

### ---------------------------------------------------------------------------------------------------------- ###
### ------------------------------------------ CONSTRAINER --------------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

class Constrainer:
    def __init__(self, load=None, sampler=None, parameters=None, c_params=None, verbose=True, df=None, 
                 nss_cube=None, sc_cube=None, a_cube=None, j_cube=None, **kwargs):
        self.sampler = sampler
        self.df = df
        
        self.nss_cube = nss_cube
        self.sc_cube = sc_cube
        self.a_cube = a_cube
        self.j_cube = j_cube
        
        # "static" parameters that we aren't constraining
        if parameters is None:
            self.parameters = dict()
        else:
            self.parameters = parameters
        
        #  parameters that we want to constrain
        if c_params is None:
            self.c_params = dict()
        else:
            self.c_params = c_params
        
        for kwarg in kwargs.keys():
            if kwargs[kwarg] == "df":
                temp_kwarg = kwarg
                if kwarg == "m1":
                    temp_kwarg = "iso_mass"
                self.parameters[kwarg] = df[temp_kwarg]
            elif isinstance(kwargs[kwarg], dict):
                self.c_params[kwarg] = kwargs[kwarg]
            else:
                self.parameters[kwarg] = kwargs[kwarg]
        
        self.verbose = verbose
        
        if load is not None:
            self.load(load)
    
    ### --- ###
    def add_parameter(self, *a, **t):
        if t is None:
            self.parameters[a[0]] = a[1]
            return
        allowed_set = ["name", "prior", "value"]
        for key in t.keys():
            if key not in allowed_set:
                raise KeyError("{} not valid key. valid options are: {}".format(key, allowed_set))
        
        self.c_params[t["name"]] = {"prior":t["prior"], "value":t["value"]}
    
    ### --- ###
    def set_initial_parameters(self, nwalkers, variation=0.3):
        parameter_names = self.c_params.keys()
        param_set = np.zeros((nwalkers, len(parameter_names)))
        
        for i, param in enumerate(parameter_names):
            prior = self.c_params[param]["prior"]
            if "value" in self.c_params[param].keys(): # randomly sampled around a given point
                value = self.c_params[param]["value"]
                param_set[:,i] = np.random.normal(loc=value, scale=value*variation, size=nwalkers)
            else: # sample uniformly in prior
                #param_set[:,i] = np.random.rand(nwalkers)*(prior[1]-prior[0]) + prior[0]
                prior_size = prior[1] - prior[0]
                values = np.linspace(prior[0]+0.05*prior_size, prior[1]-0.05*prior_size, nwalkers)
                np.random.shuffle(values)
                param_set[:,i] = values
        return param_set
        
    ### --- ###
    def within_prior(self, mcmc_params):
        for param in mcmc_params.keys():
            target_range = self.c_params[param]["prior"]
            if not within_range(mcmc_params[param], target_range):
                return False
        return True
    
    ### --- ###
    def set_initial_parameters(self, nwalkers, variation=0.3):
        parameter_names = self.c_params.keys()
        param_set = np.zeros((nwalkers, len(parameter_names)))
        
        for i, param in enumerate(parameter_names):
            prior = self.c_params[param]["prior"]
            if "value" in self.c_params[param].keys(): # randomly sampled around a given point
                value = self.c_params[param]["value"]
                param_set[:,i] = np.random.normal(loc=value, scale=value*variation, size=nwalkers)
            else: # sample uniformly in prior
                #param_set[:,i] = np.random.rand(nwalkers)*(prior[1]-prior[0]) + prior[0]
                prior_size = prior[1] - prior[0]
                values = np.linspace(prior[0]+0.05*prior_size, prior[1]-0.05*prior_size, nwalkers)
                np.random.shuffle(values)
                param_set[:,i] = values
        return param_set
    
    ### --- ###
    def run_mcmc(self, function, step_count, nwalkers, variation=0.3, args=None, kwargs=None):
        initial_params = self.set_initial_parameters(nwalkers, variation=variation)
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, len(self.c_params), function, 
                                        args=args, 
                                        kwargs=kwargs,
                                        parameter_names=list(self.c_params.keys()), pool=pool)
            sampler.run_mcmc(initial_params, step_count, progress=True, skip_initial_state_check=True)
        self.sampler = sampler
    
    ### --- ###
    def chain(self, discard=25):
        return self.sampler.get_chain(discard=discard, flat=True)
    
    ### --- ###
    def likelihoods(self, discard=25):
        return self.sampler.get_log_prob(discard=discard, flat=True)       
    
    ### --- ###
    def apply_condition(self, condition, chain, likelihoods):
        param_names = list(self.c_params.keys())
        locals = {param_names[i]:chain[:,i] for i in range(len(param_names))}
        locals["likelihood"] = likelihoods
        ran_condition = eval(condition, locals)
        
        chain = chain[ran_condition]
        total = len(likelihoods)
        likelihoods = likelihoods[ran_condition]
        if self.verbose:
            print("{}/{} ({:.1f}%) of sampled points remain".format(len(likelihoods), total, len(likelihoods)/total*100))
        
        return chain, likelihoods
        
    ### --- ###
    def plot_corner(self, condition=None, discard=25, **kwargs):
        chain, likelihoods = self.chain(discard=discard), self.likelihoods(discard=discard)
        
        if condition is not None:
            chain, likelihoods = self.apply_condition(condition, chain, likelihoods)
            
        ranges = [(param.min(), param.max()) if np.ptp(param) > 0 else (param[0]-1e-3, param[0]+1e-3)
          for param in chain.T]

        return corner.corner(chain, range=ranges, labels=list(self.c_params.keys()), **kwargs);
    
    ### --- ###
    def plot_2d(self, parameters, condition=None, truths=None, savedir=None, discard=25, **kwargs):
        chain, likelihoods = self.chain(discard=discard), self.likelihoods(discard=discard)
        
        if condition is not None:
            chain, likelihoods = self.apply_condition(condition, chain, likelihoods)

        param_names = list(self.c_params.keys())
        check_indices = [param_names.index(param) for param in parameters]

        fig, ax = plt.subplots(1,1)
        
        cb = ax.scatter(chain[:,check_indices[0]], chain[:,check_indices[1]], c=likelihoods, cmap='viridis', norm=colors.Normalize(), **kwargs)
        plt.colorbar(cb, label="log likelihood")
        ax.set_xlabel(param_names[check_indices[0]]);
        ax.set_ylabel(param_names[check_indices[1]]);
        max_x, max_y = chain[:,check_indices[0]][np.argmax(likelihoods)], chain[:,check_indices[1]][np.argmax(likelihoods)]
        ax.axvline(max_x, c="k", linestyle="--", label="highest likelihood");
        ax.axhline(max_y, c="k", linestyle="--");
        
        if truths is not None:
            ax.axvline(truths[0], c="r", linestyle="--", label="truth");
            ax.axhline(truths[1], c="r", linestyle="--");

        ax.legend()
        
        if savedir is not None:
            plt.savefig(savedir)
            
        return fig
    
    ### --- ###
    def plot_parameter(self, parameter, truth=None, condition=None, discard=25, **kwargs):
        chain, likelihoods = self.chain(discard=discard), self.likelihoods(discard=discard)
        
        if condition is not None:
            chain, likelihoods = self.apply_condition(condition, chain, likelihoods)
        
        check_index = list(self.c_params.keys()).index(parameter)
        
        plt.plot(chain[:,check_index], color="maroon")
        if truth is not None:
            plt.axhline(y=truth, c="k", linestyle="--")
        plt.title(parameter)
    
    ### --- ###
    def save_results(self, save_dir=None, name=None, note=None):
        if name == None:
            name = "binary{}".format(datetime.date.today())
            
        outdata = dict()
        outdata["metaparams"] = dict()
        outdata["metaparams"]["name"] = name
        outdata["metaparams"]["notes"] = note
        outdata["metaparams"]["timestamp"] = datetime.datetime.now()
        
        outdata["sampler"] = self.sampler
        outdata["parameters"] = self.parameters
        outdata["c_params"] = self.c_params
        
        if save_dir is None:
            save_dir = "."
            
        outfile = open("{}/{}.pkl".format(save_dir, name), "wb")
        pickle.dump(outdata, outfile)
        outfile.close()
    
    ### --- ###
    def load(self, dir):
        loaded_object = pickle.load(open(dir, "rb"))
        self.sampler = loaded_object["sampler"]
        self.parameters = loaded_object["parameters"]
        self.c_params = loaded_object["c_params"]

### --- CONSTRAINER --- ###
### ------------------- ###

### -------------- ###
### --- BINARY --- ###

class Binary(Constrainer):
    def __init__(self, df=None, catalogue_position=False, verbose=True, 
                    load=None, sampler=None, parameters=None, c_params=None, 
                    nss_cube=None, sc_cube=None, a_cube=None, j_cube=None, **kwargs):
        super().__init__(df=df, verbose=verbose, 
                          load=load, sampler=sampler, parameters=parameters, c_params=c_params, 
                          nss_cube=nss_cube, sc_cube=sc_cube, a_cube=a_cube, j_cube=j_cube, **kwargs)

        if catalogue_position:
            self.parameters["ra"] = df["ra"]
            self.parameters["dec"] = df["dec"]
            self.parameters["parallax"] = df["parallax"]
    
    ### --- ###
    def orbit_probability(self, mcmc_params, target_object, ruwe_sample_count=100, soltype_sample_count=100, bin_width=0.05, skip_soltype=True, skip_accel=True, cutoff=np.exp(-21), **kwargs):
        # make sure we're within the prior
        if not self.within_prior(mcmc_params):
            return -np.inf
        
        # we want it to vary on log scale but gaiamock wants linear scale
        if "period" in mcmc_params.keys():
            mcmc_params["period"] = 10**mcmc_params["period"]
        
        # compute distribution of ruwes
        ruwes = gw.marginalize_ruwe(sample_count=ruwe_sample_count, verbose=False, df=target_object,
                                    **mcmc_params, **kwargs)
        l_ruwe = binned_probability(np.log10(ruwes), np.log10(target_object["ruwe"]), bin_width)
        
        if skip_soltype:
            return np.log(np.maximum(l_ruwe, cutoff))
        
        solution_types = gw.marginalize_solution_type(skip_full=True, sample_count=soltype_sample_count, verbose=False, df=target_object,
                                    w=gw.random_angle, omega=gw.random_angle, **mcmc_params, **kwargs)      
        l_solution_type = len(np.where(solution_types == target_object["solution_type"])[0])/len(solution_types)
        
        if target_object["solution_type"] not in [7,9] or skip_accel:
            return np.log(np.maximum(l_ruwe*l_solution_type, cutoff))
        
        arg = "Acceleration7"
        arg1 = "acceleration7"
        if target_object["solution_type"] == 9:
            arg = "Jerk9"
            arg1 = "jerk"
        signals = gw.marginalize(gw.determine_accel, func_params=[arg], sample_count=ruwe_sample_count, verbose=False, df=target_object,
                                w=gw.random_angle, omega=gw.random_angle, **mcmc_params, **kwargs)
        l_signal = binned_probability(signals, target_object[arg1], bin_width)
        
        return np.log(np.maximum(l_ruwe*l_solution_type*l_signal, cutoff))
    
    ### --- ###
    def constrain_parameters(self, step_count=30, nwalkers=10, variation=0.3,
                             ruwe_sample_count=100, soltype_sample_count=100, bin_width=0.05, skip_soltype=True, skip_accel=True, cutoff=np.exp(-21)):
        temp_kwargs = self.parameters
        temp_kwargs["cutoff"] = cutoff
        temp_kwargs["skip_soltype"] = skip_soltype
        temp_kwargs["skip_accel"] = skip_accel
        temp_kwargs["ruwe_sample_count"] = ruwe_sample_count
        temp_kwargs["soltype_sample_count"] = soltype_sample_count
        temp_kwargs["bin_width"] = bin_width
        self.run_mcmc(self.orbit_probability, step_count, nwalkers, variation, args=(self.df,), kwargs=temp_kwargs)

### --- BINARY --- ###
### -------------- ###

### ----------------- ###
### --- CATALOGUE --- ###sc_cube, a_cube, j_cube, 

class Catalogue(Constrainer):
    def __init__(self, nss_cube=None, sc_cube=None, a_cube=None, j_cube=None, # you basically always need these
                    ps=None, qs=None, es=None, catalogue=None, verbose=True, 
                    load=None, sampler=None, parameters=None, c_params=None, **kwargs): 
        self.ps = ps
        self.qs = qs
        self.es = es
        
        super().__init__(verbose=verbose, 
                          load=load, sampler=sampler, parameters=parameters, c_params=c_params, 
                          nss_cube=nss_cube, sc_cube=sc_cube, a_cube=a_cube, j_cube=j_cube, **kwargs)

        self.catalogue = catalogue
        self.initial_params = None

    ### --- ###
    def calculate_total_likelihood(self, mcmc_params, objects, nss_cube, sc_cube, a_cube, j_cube, ps, qs, es, 
                                   ignore_orbital_solutions=False, cutoff=np.exp(-15), skip_prior=False): #, return_gaussian=False):
        ''' this function calculates the total probability of getting
            a particular set of observations by summing up the likelihoods
            for every object based on object_likelihood()
            
            This is intended to be called by the mcmc EnsembleSampler
            
            mcmc_params: the vector in parameter space that emcee throws into the function
            mcmc_hyperparams: that array of dictionaries *params from constrain_model() which contains the priors
            default_param_set: the hard coded default values for all the ones we aren't varying
        '''        
        ## make sure we're within the prior
        if not skip_prior:
            if not self.within_prior(mcmc_params):
                return -np.inf
        
        all_params = mcmc_params | self.parameters
        # calculate the model slice for this parameter set
        model_cube = calculate_model_cube(ps, qs, es, (all_params["p_mu"], all_params["p_si"]), all_params["e_g"]) 
        
        # calculate the likelihood with this parameter set
        likelihood = 0
        for target_object in objects:
            likelihood += np.log(np.maximum(catalogue_object_likelihood(target_object, model_cube, nss_cube, sc_cube, a_cube, j_cube, all_params["fb"], (all_params["ss_mu"], all_params["ss_si"]), es, ignore_orbital_solutions=ignore_orbital_solutions), cutoff))
            
        return likelihood
    
    ### --- ###
    def constrain_parameters(self, ignore_orbital_solutions=False, step_count=30, nwalkers=10, variation=0.3, cutoff=np.exp(-15)):
        temp_kwargs = dict()
        temp_kwargs["cutoff"] = cutoff
        temp_kwargs["ignore_orbital_solutions"] = ignore_orbital_solutions
        args = (self.catalogue, self.nss_cube, self.sc_cube, self.a_cube, self.j_cube, self.ps, self.qs, self.es)
        self.run_mcmc(self.calculate_total_likelihood, step_count, nwalkers, variation, args=args, kwargs=temp_kwargs)
    
    ### --- ###
    def search(self, parameter_name, value=None):
        if value is None:
            return np.array([sc[parameter_name] for sc in self.catalogue])
        return self.catalogue[[sc[parameter_name] == value for sc in self.catalogue]]
    
    ### --- ###
    def statistics(self, return_type="dict", synthetic=True):
        binary_fraction = len(self.search("is_binary", True))
        ruwes = self.search("ruwe")
        near_ss = len(self.search("solution_type", 5))
        accel = len(self.search("solution_type", 7))
        jerk = len(self.search("solution_type", 9))
        orbit = len(self.search("solution_type", 12))
        
        if return_type=="dict":
            return {"fb":binary_fraction, "p5":near_ss, "p7":accel, "p9":jerk, "p12":orbit, "ruwes":ruwes}
        else:
            return binary_fraction, near_ss, accel, jerk, orbit,
        
        
### --- CATALOGUE --- ###
### ----------------- ###