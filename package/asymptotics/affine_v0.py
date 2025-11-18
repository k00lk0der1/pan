from .utils import * 

def affine_V_0(alpha, tolerance, L_max_start):
    #For affine case (known value of beta=1)
    beta=1
    lambda_val, L_max = find_malthusian_parameter(tolerance, alpha, beta, L_max_start)

    k = np.arange(1, L_max+1)
    f_val = f(k, alpha, beta)

    p_k_val = p_k(f_val, lambda_val)

    p_greater_k_val = p_greater_k(f_val, p_k_val, lambda_val)

    f_grad_val_norm = np.ones(shape=f_val.shape)/f_val
    #Gradient of f wrt alpha=1 for all values of alpha and k when beta=1

    ft_mat = (
        f_grad_val_norm.reshape(-1, 1, L_max) * 
        f_grad_val_norm.reshape(1, -1, L_max) * 
        p_greater_k_val
    ).sum(axis=-1)


    st_vec = (f_grad_val_norm*p_greater_k_val).sum(axis=0, keepdims=True)
    st_mat = st_vec@st_vec.T

    V_0_val = ft_mat - st_mat

    return V_0_val
