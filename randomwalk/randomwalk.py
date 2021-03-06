import numpy as np

def mom_random_walk(T,F,G,dx,dv,initx = 0,initv = 0):
    
    '''
    x_t+1 = F.x_t + v_t + dx
    v_t+1 = G.v_t + dv
    
    dx and dv are NOISE
    '''
    
    DX = np.random.multivariate_normal(np.zeros(F.shape[0]),dx,size = [T])
    DV = np.random.multivariate_normal(np.zeros(F.shape[0]),dv,size = [T])
    
    if initx == 0:
        outx = np.zeros([1,F.shape[0]])
    else:
        outx = np.array([initx])
        
    if initv == 0:
        outv = np.zeros([1,F.shape[0]])
    else:
        outv = np.array([initv])
        
    for k in range(len(DX)):
        outv = np.append(outv,np.dot(G,outv[-1]) + DV[k])
        outx = np.append(outx,np.dot(F,outx[-1]) + outv[-1] + DX[k])
        
    return outx, outv

def gauss_random_walk(dx_var, F, T, init = 0):
    '''
    Description: generates a gaussian random walk with specified statistics and number of time steps
    
    args:
     dx_var - variance of steps
     F  - the matrix determining the linear dynamics of the thing
     T - number of time steps
     init - where to start the walk
     
    return:
     ndarray - results of the walk
    '''
    
    temp = np.random.multivariate_normal(np.zeros(dx_var.shape[0]),dx_var,size = [T])
    
    if init == 0:
        out = np.zeros([1,dx_var.shape[0]])
    else:
        out = np.array([init])
        
    for k in temp:
        out = np.append(out,[np.dot(F,out[-1]) + k],axis = 0)

    return out

def globvar_gauss_random_walk(T,tau,var):
    
    '''
    description: generates a gaussian random walk with a specified variance and correlation timescale
    
    args:
     T - number of timesteps
     tau - timescale 
     var - variance of the walk

    returns
     ndarray - result
    
    notes:
     The equation relating the variance V to individual step variance S and linear dynamics F is V = F.V.F + S. If we speficy V and F then S is easy to solve for: S = V - F.V.F. We, on the other hand, have specified tau, which is somewhat ambiguous for multidimentional random walks.
    '''
    
    
