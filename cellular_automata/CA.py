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
        
    def set_p(self,p):
        self.rng = lambda :np.random.multinomial(1,[1.-p,p],[size])[:,1]        
        
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
        self.p = p
        self.rule = rule
        self.size = size
        self.rng = lambda :np.random.multinomial(1,[1.-p,p],[size,size])[:,:,1]        
        self.rst = lambda :np.random.randint(0,2,[size,size])
        self.name = name
        self.nbhd = nbhd
        
    def rand_init(self,r = .5):
        self.state = np.random.multinomial(1,[1.-r,r],[self.size,self.size])[:,:,1]
        
    def pentomino_init(self):
        self.state = np.zeros_like(self.state)
            
        n = int(self.size/2)
        
        pnt = [[n,n],[n+1,n],[n-1,n],[n-1,n-1],[n,n+1]]
        
        for p in pnt:   
            self.state[p[0],p[1]] = 1
        
    def set_p(self,p):
        self.p = p
        self.rng = lambda :np.random.multinomial(1,[1.-p,p],[self.size,self.size])[:,:,1]        
        
    def update(self,ns = 1,PRINT = False):
        for k in range(ns):

            conv = np.int32(np.rint(scipy.signal.fftconvolve(self.state,self.nbhd,mode = 'same')))

            temp = np.copy(self.state)

            m_01 = [conv == i for i in range(len(self.rule[0])) if self.rule[0][i] == 1]
            m_11 = [conv == i for i in range(len(self.rule[1])) if self.rule[1][i] == 1]
            
            mask_01 = m_01[0]
            mask_11 = m_11[0]
            
            for m in m_01[1:]:
                mask_01 += m
            for m in m_11[1:]:
                mask_11 += m
                
            self.state[(temp == 0)*(mask_01)] = 1
            self.state[(temp == 1)*(mask_11)] = 1
            
            self.state[(temp == 0)*(mask_01)] = 1
            self.state[(temp == 1)*(mask_11)] = 1

            
            self.state[(temp == 0)*np.logical_not(mask_01)] = 0
            self.state[(temp == 1)*np.logical_not(mask_11)] = 0

            num = self.rng()
            rst = self.rst()
            
            self.state = self.state*(1 - num) + (rst)*num

            if PRINT:
                print(self.state)
            return self.state

    def chunk_state(self,nchunk):
        ch = np.ones([nchunk,nchunk])
        return (scipy.signal.fftconvolve(self.state,ch,"same")/np.float32(nchunk**2))[::nchunk,::nchunk]
    
    def get_data(self,niter,transient = 0,r = .5):
            self.rand_init(r = r)
            out = []
            
            for k in range(niter + transient):
                self.update()
                
                out.append(self.state)
                
            out = np.array(out)
            
            return out[transient:]
class tf_CA2D:
    def __init__(self,size,rule,iters = 1,p = .01,name = "",nbhd = np.array([[1,1,1],[1,0,1],[1,1,1]])):
        self.state = tf.Variable(np.float32(np.zeros([iters,size,size])),name = name + "_state")
        self.p = tf.Variable(np.float32(p),name = name + "_prob")
        self.rule = rule
        self.size = size
        self.rng = lambda :np.random.multinomial(1,[1.-p,p],[size,size])[:,:,1]        
        self.rst = lambda :np.random.randint(0,2,[size,size])
        self.name = name
        self.nbhd = nbhd
        
    def rand_init(self,r = .5):
        self.state = np.random.multinomial(1,[1.-r,r],[self.size,self.size])[:,:,1]
        
    def pentomino_init(self):
        self.state = np.zeros_like(self.state)
            
        n = int(self.size/2)
        
        pnt = [[n,n],[n+1,n],[n-1,n],[n-1,n-1],[n,n+1]]
        
        for p in pnt:   
            self.state[p[0],p[1]] = 1
        
    def set_p(self,p):
        self.p = p
        self.rng = lambda :np.random.multinomial(1,[1.-p,p],[self.size,self.size])[:,:,1]        
        
    def update(self,ns = 1,PRINT = False):
        for k in range(ns):

            conv = np.int32(np.rint(scipy.signal.fftconvolve(self.state,self.nbhd,mode = 'same')))

            temp = np.copy(self.state)

            m_01 = [conv == i for i in range(len(self.rule[0])) if self.rule[0][i] == 1]
            m_11 = [conv == i for i in range(len(self.rule[1])) if self.rule[1][i] == 1]
            
            mask_01 = m_01[0]
            mask_11 = m_11[0]
            
            for m in m_01[1:]:
                mask_01 += m
            for m in m_11[1:]:
                mask_11 += m
                
            self.state[(temp == 0)*(mask_01)] = 1
            self.state[(temp == 1)*(mask_11)] = 1
            
            self.state[(temp == 0)*(mask_01)] = 1
            self.state[(temp == 1)*(mask_11)] = 1

            
            self.state[(temp == 0)*np.logical_not(mask_01)] = 0
            self.state[(temp == 1)*np.logical_not(mask_11)] = 0

            num = self.rng()
            rst = self.rst()
            
            self.state = self.state*(1 - num) + (rst)*num

            if PRINT:
                print(self.state)
            return self.state

    def chunk_state(self,nchunk):
        ch = np.ones([nchunk,nchunk])
        return (np.convolve(self.state,ch,"same")/np.float32(nchunk))[::nchunk]
    
    def get_data(self,niter,transient = 0,r = .5):
            self.rand_init(r = r)
            out = []
            
            for k in range(niter + transient):
                self.update()
                
                out.append(self.state)
                
            out = np.array(out)
            
            return out[transient:]
                
if __name__=="__main__":
    
    ca = CA2D(10,[[0,0,0,1,1,0,0,0],[0,0,0,1,0,0,0,0]])

    out = []

    t1 = time.time()
    
    for k in range(1000):
        ca.update(1)
        out.append(ca.state)
    print(time.time() - t1)
        
    np.savetxt("./test.csv",np.reshape(np.array(out),[-1,10*10]))
