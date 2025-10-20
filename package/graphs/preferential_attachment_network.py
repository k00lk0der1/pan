import copy
import numpy as np

#import jax
#import jax.numpy as jnp

#import pytensor.tensor as pt 
#import pymc as pm


class PreferentialAttachmentNetwork:
    def __init__(self):
        self.observed = False
    
    def generate_sample(self, alpha, beta, n_nodes, random_state_seed=None):
        
        #Setting the seed and generating a RandomState object.
        self.random_state_seed = random_state_seed
        random_state = np.random.RandomState(self.random_state_seed)
        
        #Generating samples from the uniform distribution to use for inverse CDF tranform
        uniform_samples = random_state.uniform(size=(n_nodes-2))

        #Initializing Graph State
        self.degrees = np.zeros(shape=(n_nodes))
        self.degrees[:2] = 1

        self.d_t = np.zeros(shape=(n_nodes))

        f = lambda x : np.power(x + alpha, beta)
        
        for t in range(2, n_nodes):
            parent_node_probability_propto = f(self.degrees[:t])
            
            parent_node_probability_cdf = (
                parent_node_probability_propto.cumsum() / 
                parent_node_probability_propto.sum()
            )

            chosen_node_index = (parent_node_probability_cdf<uniform_samples[t-2]).argmax()

            self.d_t[t] = self.degrees[chosen_node_index]
            self.degrees[t] = 1
            self.degrees[chosen_node_index] = self.degrees[chosen_node_index] + 1
        
        N_dict = {
            2:{1:2}
        }

        for t in range(3, n_nodes):
            N_dict[t] = copy.deepcopy(N_dict[t-1])
            N_dict[t][1] = N_dict[t][1] + 1
            N_dict[t][self.d_t[t-1]] = N_dict[t][self.d_t[t-1]] - 1
            if self.d_t[t-1]+1 in N_dict[t].keys():
                N_dict[t][self.d_t[t-1]+1] = N_dict[t][self.d_t[t-1]] - 1
            else:
                N_dict[t][self.d_t[t-1]+1] = 1
            
            if(N_dict[t][self.d_t[t-1]]==0):
                N_dict[t].pop(self.d_t[t-1])
            
            print(N_dict)
            
        
        self.observed = True
                    

    def negative_log_likelihood(self, alpha, beta, n_nodes):
        pass
    
    def numerical_mle(self, n_nodes):
        pass
    
    def generate_posterior_samples(self, alpha_prior_factory, beta_prior_factory, n_nodes):
        pass