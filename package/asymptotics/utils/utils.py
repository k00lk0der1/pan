import numpy as np
import scipy.optimize

def f(k, alpha, beta):
    return np.power(k+alpha, beta)

def f_grad(k, alpha, beta):
    return np.array(
        [
            beta*np.power(k+alpha, beta-1),
            np.power(k+alpha, beta) * np.log(k+alpha)
        ]
    )

def g(lambda_val, alpha, beta, L_max):
    f_val = f(np.arange(1, L_max+1), alpha, beta)
    return (f_val/(lambda_val+f_val)).cumprod().sum()-1

def find_malthusian_parameter(tolerance, alpha, beta, L_max_start):
    L_max = L_max_start
    
    current_sol_root = np.nan
    new_sol_root = np.nan
    
    while(np.abs(current_sol_root - new_sol_root)>tolerance or np.isnan(current_sol_root)):
        current_sol_root = new_sol_root
        
        new_solution = scipy.optimize.root_scalar(
            g,
            args=(alpha, beta, L_max),
            method='brentq',
            bracket=[0.001, 1000]
        )
        
        if(new_solution.converged):
            new_sol_root = new_solution.root
        
        L_max = L_max*10
    
    return new_sol_root,  L_max//10

def p_k(f_val, lambda_val):
    
    terms = np.concatenate(
        [
            [1.0],
            f_val/(lambda_val+f_val)
        ]
    ).cumprod()
    
    p_k_val = ((terms[:-1]*lambda_val)/(lambda_val + f_val))
    
    return p_k_val

def p_greater_k(f_val, p_k_val, lambda_val):
    return (f_val*p_k_val)/lambda_val