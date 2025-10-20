import copy
import numpy as np
import tqdm

#import jax
#import jax.numpy as jnp

#import pytensor.tensor as pt 
#import pymc as pm


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
            (np.power(self.d_t+alpha, beta)*self.N_t_d_t)/
            (np.power(self.self.N_deg+alpha, beta) * self.N).sum(axis=1)
        )
        return (-log_lik)
    
    def numerical_mle(self, n_nodes):
        pass
    
    def generate_posterior_samples(self, alpha_prior_factory, beta_prior_factory, n_nodes):
        pass