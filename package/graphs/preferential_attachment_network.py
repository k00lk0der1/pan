import copy
import numpy as np
import tqdm
import pytensor.tensor as pt
import pytensor
import scipy.stats, scipy.optimize 
import pymc as pm
import numpyro


class PreferentialAttachmentNetwork:
    def __init__(self):
        self.observed = False
    
    def generate_sample(self, alpha, beta, n_nodes, random_state_seed=None, disable_progress_bar=True):
        
        #Setting the seed and generating a RandomState object.
        self.random_state_seed = random_state_seed
        random_state = np.random.RandomState(self.random_state_seed)
        
        #Generating samples from the uniform distribution to use for inverse CDF tranform
        uniform_samples = random_state.uniform(size=(n_nodes-2))

        #Initializing Graph State
        self.degrees = np.zeros(shape=(n_nodes)).astype(np.int64)
        self.degrees[:2] = 1

        self.d_t = np.zeros(shape=(n_nodes)).astype(np.int64)
        self.N_t_d_t = np.zeros(shape=(n_nodes)).astype(np.int64)

        f = lambda x : np.power(x + alpha, beta)
        
        for t in tqdm.tqdm(range(2, n_nodes), disable=disable_progress_bar):
            parent_node_probability_propto = f(self.degrees[:t])
            
            parent_node_probability_cdf = (
                parent_node_probability_propto.cumsum() / 
                parent_node_probability_propto.sum()
            )

            chosen_node_index = (parent_node_probability_cdf>uniform_samples[t-2]).argmax()
            

            self.d_t[t] = self.degrees[chosen_node_index]
            self.degrees[t] = 1
            self.degrees[chosen_node_index] = self.degrees[chosen_node_index] + 1

        
        N_dict = {
            2:{np.int64(1):2}
        }
        
        self.N_t_d_t[2] = np.int64(2)

        max_unique_degree = 2

        for t in tqdm.tqdm(range(3, n_nodes), disable=disable_progress_bar):

            N_dict[t] = copy.deepcopy(N_dict[t-1])
            N_dict[t][np.int64(1)] = N_dict[t][np.int64(1)] + 1
            N_dict[t][self.d_t[t-1]] = N_dict[t][self.d_t[t-1]] - 1

            if self.d_t[t-1]+1 in N_dict[t].keys():
                N_dict[t][self.d_t[t-1]+1] = N_dict[t][self.d_t[t-1]+1] + 1
            else:
                N_dict[t][self.d_t[t-1]+1] = 1
            
            if(N_dict[t][self.d_t[t-1]]==0):
                N_dict[t].pop(self.d_t[t-1])
            
            self.N_t_d_t[t] = N_dict[t][self.d_t[t]]
            
            max_unique_degree = max(
                max_unique_degree,
                len(N_dict[t].keys())
            )
        
        self.N = np.zeros(shape=(n_nodes-2, max_unique_degree))
        self.N_deg = np.zeros(shape=(n_nodes-2, max_unique_degree))

        for t in tqdm.tqdm(range(2, n_nodes), disable=disable_progress_bar):
            for degree_counter, d in enumerate(N_dict[t].keys()):
                self.N[t-2][degree_counter] = N_dict[t][d]
                self.N_deg[t-2][degree_counter] = d
        
        self.d_t =  self.d_t[2:]
        self.N_t_d_t =  self.N_t_d_t[2:]
        
        self.observed = True
                    

    def negative_log_likelihood(self, alpha, beta, n_nodes):
        
        log_lik = (
            (beta*pt.log(self.d_t[:n_nodes-2]+alpha)) + 
            pt.log(self.N_t_d_t[:n_nodes-2]) -
            pt.log(
                (pt.power(self.N_deg[:n_nodes-2]+alpha, beta) * self.N[:n_nodes-2]).sum(axis=1)
            )
        ).sum()
    
        return (-log_lik)
    
    def numerical_mle(self, init_alpha_guess, init_beta_guess, n_nodes):
        alpha_sym = pt.scalar('alpha')
        beta_sym = pt.scalar('beta')

        cost = self.negative_log_likelihood(alpha_sym, beta_sym, n_nodes)

        grad_cost = pt.grad(cost, wrt=[alpha_sym, beta_sym])

        f_cost_grad = pytensor.function(
            inputs=[alpha_sym, beta_sym],
            outputs=[cost, pt.stack(grad_cost)]
        )

        def objective_with_grad(params):
            alpha_val, beta_val = params
            cost_val, grad_val = f_cost_grad(alpha_val, beta_val)
            return cost_val, grad_val.astype('float64')

        initial_params = np.array([init_alpha_guess, init_beta_guess])

        result = scipy.optimize.minimize(
            fun=objective_with_grad,
            x0=initial_params,
            bounds=[
                (0, np.inf),
                (0,1)
            ],
            method='L-BFGS-B',
            jac=True
        )

        return result
    
    def generate_posterior_samples(
        self,
        alpha_prior_factory,
        beta_prior_factory,
        n_nodessamples = 10000,
        warmup = 5000,
        chains = 4,
        cores = 4,
        nuts_sampler = "numpyro",
        progressbar=False
    ):
        pan_model = pm.Model()
        
        with pan_model as model:
            alpha = alpha_prior_factory()
            beta = beta_prior_factory()
            likelihood = pm.Potential(
                'likelihood',
                -self.negative_log_likelihood(
                    alpha,
                    beta,
                    n_nodes
                )
            )

        idata = pm.sample(
            draws=samples,
            tune=warmup,
            chains=chains,
            cores=cores,
            model=pan_model,
            nuts_sampler=nuts_sampler,
            progressbar=progressbar
        )

        return idata
        