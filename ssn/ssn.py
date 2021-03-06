import numpy as np

class SSN:
    def __init__(self,W,k,tau,n,dt = .001,init = 0):
        self.N = len(W)
        self.W = W
        self.k = k
        self.tau = tau        
        self.n = n

        self.dt = dt

        if type(init) == np.ndarray:
            self.init = init
        else:
            self.init = np.zeros(self.N)

    def get_rates(self,I):
        return run_to_fixed_point(self,I,init = self.init,dt = self.dt)
        

def rec(x):
    return (np.abs(x) + x)/2
    
def rectified_power(x,n):
    return np.power(rec(x),n)

def drdt_f(r,I,W,k,tau,n):
    return (-r + k*rectified_power(np.dot(W,r)+I,n))/tau

def run_to_fixed_point(SSN,I,init = 0,dt = .003):
    if type(init) == np.ndarray:
        r = np.array([init])
    else:
        r = np.zeros([1,SSN.N])
        
    drdt = drdt_f(r[-1],I,SSN.W,SSN.k,SSN.tau,SSN.n)
    
    t = 0

    while np.any(np.abs(drdt) > 1e-10) and t < 10000*dt:
        r = np.append(r,rec(np.array([r[-1] + drdt * dt])),axis = 0)

        t += dt

        drdt = drdt_f(r[-1],I,SSN.W,SSN.k,SSN.tau,SSN.n)

    return r,t,drdt

if __name__ == "__main__":
    n = 2.2
    k = .01
    W = np.array([[1.5,-1.3],[1.1,-.5]])
    tau = np.array([.01,.002])
    ssn = SSN(W,k,tau,2.2)    
    I = np.array([20,20])
    
    r,t,drdt = run_to_fixed_point(ssn,I)
    r2,t2,drdt2 = ssn.get_rates(I)

    print(r.shape)
    
    print(r[-1])
