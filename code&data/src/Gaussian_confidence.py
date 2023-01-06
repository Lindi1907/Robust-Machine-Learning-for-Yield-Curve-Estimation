import numpy as np


def KR_gaussian(C,
        B,
        ridge,
        inv_w,
        K
       ):

    '''
    - Get solution in KR model
    - Args: 
        - C (numpy array of dim (nt, Nmax)): cashflow matrix
        - B (numpy array of dim (nt,)): price vector corresponding to C
        - ridge (float): the ridge hyper-parameter. Require ridge>=0
        - inv_w (numpy array of dim (nt,)): inverse of weight vector w
        - K (numpy array of dim (Nmax, Nmax_y)): kernel matrix specific to kernel hyper-parameter alpha and delta.
            Nmax_y (in days) is the limit of extrapolation. 
    - Returns:
        - var (numpy array of dim (Nmax,)): solved variance
        - std (numpy array of dim (Nmax,)): solved standard deviation
        - lower95 (): 95% confidence interval of fitted price, lower
        - upper95 (): 95% confidence interval of fitted price, upper
    '''

    Nmax,Nmax_y=K.shape
    nt=B.shape[0]

    # get column indexes with nonzero cashflow
    arr_msk_col=np.where(C.sum(axis=0)!=0)[0]
    # max time to nonzero cashflow (in days) 
    tau_max_inday=arr_msk_col[-1]+1
    l_scaled=ridge/tau_max_inday

    # only keep rows and columns in K corresponding to nonzero cashflow days
    K_masked=K.take(arr_msk_col,axis=0).take(arr_msk_col,axis=1)

    # only keep columns of C with cashflow
    C_masked=C[:,arr_msk_col]
    Nt=C_masked.shape[1]
    # x: vector of time to cashflow dates (in year)
    x=(arr_msk_col+1)/365
    # 
    # get solution for (beta, r)
    CKC_inv=np.linalg.inv(C_masked@K_masked@C_masked.T+l_scaled*inv_w*np.identity(nt))
    # get coefficient vector beta. shape of beta is (nt,)
    var = K - K.take(arr_msk_col,axis=1)@(C_masked.T)@CKC_inv@C_masked@K.take(arr_msk_col,axis=0)
    
    beta=(C_masked.T@CKC_inv)@(B-C_masked@np.ones(Nt))
    # get discount vector with length Nmax
    g_solved=1+K.take(arr_msk_col,axis=1)@beta
    fitted_price = C@g_solved[:C.shape[1]]
    var_price = C@var[:C.shape[1], :C.shape[1]]@C.T
    std_price = np.sqrt(var_price)
    
    var_d = np.diag(var)
    std = np.sqrt(var_d) 
    
    lower95 = fitted_price - 2*(np.diag(std_price)[:C.shape[1]])
    upper95 = fitted_price + 2*(np.diag(std_price)[:C.shape[1]])
    return var_d, std, lower95, upper95

def KR_gaussian2(C,
        B,
        ridge,
        inv_w,
        K
       ):

    '''
    - Get solution in KR model
    - Args: 
        - C (numpy array of dim (nt, Nmax)): cashflow matrix
        - B (numpy array of dim (nt,)): price vector corresponding to C
        - ridge (float): the ridge hyper-parameter. Require ridge>=0
        - inv_w (numpy array of dim (nt,)): inverse of weight vector w
        - K (numpy array of dim (Nmax, Nmax_y)): kernel matrix specific to kernel hyper-parameter alpha and delta.
            Nmax_y (in days) is the limit of extrapolation. 
    - Returns:
        - var (numpy array of dim (Nmax,)): solved variance
        - std (numpy array of dim (Nmax,)): solved standard deviation
        - lower95 (): 95% confidence interval of fitted price, lower
        - upper95 (): 95% confidence interval of fitted price, upper
    '''

    Nmax,Nmax_y=K.shape
    nt=B.shape[0]

    # get column indexes with nonzero cashflow
    arr_msk_col=np.where(C.sum(axis=0)!=0)[0]
    # max time to nonzero cashflow (in days) 
    tau_max_inday=arr_msk_col[-1]+1
    l_scaled=ridge/tau_max_inday

    # only keep rows and columns in K corresponding to nonzero cashflow days
    K_masked=K.take(arr_msk_col,axis=0).take(arr_msk_col,axis=1)

    # only keep columns of C with cashflow
    C_masked=C[:,arr_msk_col]
    Nt=C_masked.shape[1]
    # x: vector of time to cashflow dates (in year)
    x=(arr_msk_col+1)/365
    # 
    # get solution for (beta, r)
    CKC_inv=np.linalg.inv(C_masked@K_masked@C_masked.T+l_scaled*inv_w*np.identity(nt))
    # get coefficient vector beta. shape of beta is (nt,)
    var = K[:C.shape[1],:C.shape[1]] - K[:C.shape[1],:C.shape[1]]@(C.T)@CKC_inv@C@K[:C.shape[1],:C.shape[1]]
    
    beta=(C_masked.T@CKC_inv)@(B-C_masked@np.ones(Nt))
    # get discount vector with length Nmax
    g_solved=1+K.take(arr_msk_col,axis=1)@beta
    fitted_price = C@g_solved[:C.shape[1]]
    var_price = C@var[:C.shape[1], :C.shape[1]]@C.T
    std_price = np.sqrt(var_price)
    var_d = np.diag(var)
    std = np.sqrt(var_d) 
    
    lower95 = fitted_price - 2*(np.diag(std_price))
    upper95 = fitted_price + 2*(np.diag(std_price))
    return var_d, std, lower95, upper95


