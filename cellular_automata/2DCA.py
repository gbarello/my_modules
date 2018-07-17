import numpy as np
import time
import scipy.ndimage
import scipy.signal
def N_to_R(n):
    temp = n
    out = []

    for k in range(8):
        if temp >= 2**(7 - k):
            out.append(1)
            temp -= 2**(7 - k)
        else:
            out.append(0)

    return np.array(out)

def R_to_N(R):
    return sum([2**(len(R) - 1 - k)*R[k] for k in range(len(R))])

class CA:
    
    def __init__(self,size,ninst = 1,rule = N_to_R(110),name = "",p = .001,nbhd = [1,2,4]):
        
        self.state = np.int32(np.zeros(size))
        self.nbhd = np.int32(np.array(nbhd))
        self.rule = np.int32(np.array(rule))
        self.rng = lambda :np.random.multinomial(1,[1.-p,p],[size])[:,1]

        self.name = name        

    def update(self,ns = 1,PRINT = False):
        for k in range(ns):
            num = self.rng()

            self.state = self.state*(1 - num) + (1 - self.state)*num

            conv = scipy.ndimage.filters.convolve1d(self.state,self.nbhd,mode = 'wrap')

            self.state = self.rule[7 - conv]

            if PRINT:
                print(self.state)

    def chunk_state(self,nchunk):
        ch = np.ones(nchunk)
        return (np.convolve(self.state,ch,"same")/np.float32(nchunk))[::nchunk]

class CA2D:
    def __init__(self,size,rule,p = .01,name = "",nbhd = np.array([[1,1,1],[1,0,1],[1,1,1]])):
        self.state = np.int32(np.zeros([size,size]))
        self.rule = rule
        self.rng = lambda :np.random.multinomial(1,[1.-p,p],[size,size])[:,:,1]
        self.name = name
        self.nbhd = nbhd
        
    def update(self,ns = 1,PRINT = False):
        for k in range(ns):
            num = self.rng()

            self.state = self.state*(1 - num) + (1 - self.state)*num

            conv = np.array(scipy.signal.fftconvolve(self.state,self.nbhd,mode = 'same'),np.int32)

            temp = np.copy(self.state)

            self.state[(temp == 0)*((conv == 2) + (conv == 3))] = 1
            self.state[(temp == 1)*(conv == 3)] = 1
            
            self.state[(temp == 0)*(1-((conv == 2) + (conv == 3)))] = 0
            self.state[(temp == 1)*(1-(conv == 3))] = 0
            
            if PRINT:
                print(self.state)

    def chunk_state(self,nchunk):
        ch = np.ones([nchunk,nchunk])
        return (np.convolve(self.state,ch,"same")/np.float32(nchunk))[::nchunk]
    
            
if __name__=="__main__":
    
    ca = CA2D(10,[[0,0,0,1,1,0,0,0],[0,0,0,1,0,0,0,0]])

    out = []

    t1 = time.time()
    
    for k in range(1000):
        ca.update(1)
        out.append(ca.state)
    print(time.time() - t1)
        
    np.savetxt("./test.csv",np.reshape(np.array(out),[-1,10*10]))