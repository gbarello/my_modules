import theano.tensor as T
import theano
import numpy as np

def get_distribution(loss_type,params):
    if loss_type == "gauss":
        return make_G(params)
    elif loss_type == "exp":
        return make_E(params)
    elif loss_type == "cauch":
        return make_C(params)
    elif loss_type == "spikenslab":
        return make_SS(params)
    elif loss_type == "e_spikenslab":
        return make_E_SS(params)
    elif loss_type == "c_spikenslab":
        return make_C_SS(params)
    else:
        print("error! Loss function not recognized")
        exit()

def make_G(params):

    def f(latvals):
        return -(latvals**2).sum(axis = 2)/2 - (latvals.shape[2]/2)*np.log(2*np.pi)

    def g(latvals):
        return -(latvals)

    def dist(x,y):
        return np.random.randn(x,y)

    return f,g,dist
    
def make_SS(params):

    s1 = params["s1"]
    s2 = params["s2"]
    S = params["S"]
    
    #Log[P] = Log[S g1 + (1-S)g2]
    
    def f(latvals):
        D1 = -((latvals/s1)**2).sum(axis = 2)/2 - (latvals.shape[2]/2)*np.log(2*np.pi*(s1**2))
        D2 = -((latvals/s2)**2).sum(axis = 2)/2 - (latvals.shape[2]/2)*np.log(2*np.pi*(s2**2))

        return T.log(S) + D2 + T.log(1 + (1. - S)*T.exp(D1 - D2)/S)#T.log(S*T.exp(D2) + (1.-S)*T.exp(D1))

    def g(latvals):
        D1 = -((latvals/s1)**2).sum(axis = 2)/2 - (latvals.shape[2]/2)*np.log(2*np.pi*(s1**2))
        D2 = -((latvals/s2)**2).sum(axis = 2)/2 - (latvals.shape[2]/2)*np.log(2*np.pi*(s2**2))

        D1 = D1.dimshuffle([0,1,'x'])
        D2 = D2.dimshuffle([0,1,'x'])
        
        DD1 = -latvals/(s1**2)
        DD2 = -latvals/(s2**2)

        return DD2 + ((DD1-DD2)*(1. - S)*T.exp(D1 - D2)/S)/(1 + (1. - S)*T.exp(D1 - D2)/S)#(S*T.exp(D2)*DD2 + (1.-S)*T.exp(D1)*DD1)/(S*T.exp(D2) + (1.-S)*T.exp(D1))
    #D1 + (T.exp(DD2 - DD1)*(D2 - D1)*(S/(1-S)))/(1  + T.exp(DD2 - DD1)*(S/(1-S)))

    def dist(x,y):
        a = np.random.uniform(0,1,[x])

        b = np.zeros_like(a)
        
        b[a < S] = 1
        b[b < 1] = 0

        b = np.reshape(b,[-1,1])
        
        small = np.random.randn(x,y)*s1
        big = np.random.randn(x,y)*s2
        
        return big*b + small*(1-b)

    return f,g,dist

def make_E_SS(params):
    #Log[P] = Log[S g1 + (1-S)g2]
    
    s1 = params["s1"]
    s2 = params["s2"]
    S = params["S"]
    
    def f(latvals):
        D1 = -((latvals/s1)**2).sum(axis = 2)/2 - (latvals.shape[2]/2)*np.log(2*np.pi*(s1**2))
        D2 = -(T.abs_(latvals/s2)).sum(axis = 2) - (latvals.shape[2])*np.log(2*s2)

        #return T.log(S*T.exp(D2) + (1.-S)*T.exp(D1))
        return T.log(S) + D2 + T.log(1 + (1. - S)*T.exp(D1 - D2)/S)

    def g(latvals):
        D1 = -((latvals/s1)**2).sum(axis = 2)/2 - (latvals.shape[2]/2)*np.log(2*np.pi*(s1**2))
        D2 = -(T.abs_(latvals/s2)).sum(axis = 2) - (latvals.shape[2])*np.log(2*s2)

        D1 = D1.dimshuffle([0,1,'x'])
        D2 = D2.dimshuffle([0,1,'x'])
        
        DD1 = -latvals/(s1**2)
        DD2 = -T.sgn(latvals/s2)

        return DD2 + ((DD1-DD2)*(1. - S)*T.exp(D1 - D2)/S)/(1 + (1. - S)*T.exp(D1 - D2)/S)#(S*T.exp(D2)*DD2 + (1.-S)*T.exp(D1)*DD1)/(S*T.exp(D2) + (1.-S)*T.exp(D1))
    #D1 + (T.exp(DD2 - DD1)*(D2 - D1)*(S/(1-S)))/(1  + T.exp(DD2 - DD1)*(S/(1-S)))

    def dist(x,y):
        a = np.random.uniform(0,1,[x])

        b = np.zeros_like(a)
        
        b[a < S] = 1
        b[b < 1] = 0

        b = np.reshape(b,[-1,1])
        
        small = np.random.randn(x,y)*s1
        big = np.random.laplace(0,1,[x,y])*s2
        
        return big*b + small*(1-b)

    return f,g,dist

def make_C_SS(params):
    #Log[P] = Log[S g1 + (1-S)g2]
    
    s1 = params["s1"]
    s2 = params["s2"]
    S = params["S"]

    def f(latvals):
        D1 = -((latvals/s1)**2).sum(axis = 2)/2 - (latvals.shape[2]/2)*np.log(2*np.pi*(s1**2))
        D2 = -T.log(1. + (latvals/s2)**2).sum(axis = 2) - latvals.shape[2]*np.log(np.pi*s2)

        return T.log(S) + D2 + T.log(1 + (1. - S)*T.exp(D1 - D2)/S)#T.log(S*T.exp(D2) + (1.-S)*T.exp(D1))

    def g(latvals):
        D1 = -((latvals/s1)**2).sum(axis = 2)/2 - (latvals.shape[2]/2)*np.log(2*np.pi*(s1**2))
        D2 = -T.log(1. + (latvals/s2)**2).sum(axis = 2) - latvals.shape[2]*np.log(np.pi*s2)

        D1 = D1.dimshuffle([0,1,'x'])
        D2 = D2.dimshuffle([0,1,'x'])
        
        DD1 = -latvals/(s1**2)
        DD2 = -2*(latvals/(s2*s2))/(1. + (latvals/s2)**2)

        return DD2 + ((DD1-DD2)*(1. - S)*T.exp(D1 - D2)/S)/(1 + (1. - S)*T.exp(D1 - D2)/S)
        #return (S*T.exp(D2)*DD2 + (1.-S)*T.exp(D1)*DD1)/(S*T.exp(D2) + (1.-S)*T.exp(D1))
    
    #D1 + (T.exp(DD2 - DD1)*(D2 - D1)*(S/(1-S)))/(1  + T.exp(DD2 - DD1)*(S/(1-S)))

    def dist(x,y):
        a = np.random.uniform(0,1,[x])

        b = np.zeros_like(a)
        
        b[a < S] = 1
        b[b < 1] = 0

        b = np.reshape(b,[-1,1])
        
        small = np.random.randn(x,y)*s1
        big = np.random.standard_cauchy([x,y])*s2
        
        return big*b + small*(1-b)

    return f,g,dist

def make_E(parms):
    def f(latvals):
        exp = (-T.abs_(latvals)).sum(axis = 2) - (latvals.shape[2])*np.log(2)
        return exp
    def g(latvals):
        return - T.sgn(latvals)
    def dist(x,y):
        return np.random.laplace(0,1,[x,y])

    return f,g,dist
    
def make_C(parms):
    def f(latvals):
        return -T.log(1. + latvals**2).sum(axis = 2) - latvals.shape[2]*np.log(np.pi)

    def g(latvals):
        return -2*latvals/(1. + latvals**2)

    def dist(x,y):
        return np.random.standard_cauchy([x,y])
    
    return f,g,dist

