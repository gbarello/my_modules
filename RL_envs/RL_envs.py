import numpy as np

class long_walker:

    def __init__(self,length):

        self.len = length

        self.pos = np.array([0,0])

    def actions(self):
        return [np.array([0,0]),np.array([1,0]),np.array([-1,0]),np.array([0,1]),np.array([0,-1])]
    def update(self,a):
            
        if self.pos[1] == 0:
            if self.pos[0] < self.len:
                self.pos += np.array([a[0],0])
            elif a.tolist() in [[-1,0],[0,1],[0,0]]:
                self.pos += a
            
            self.pos = np.int32((self.pos + np.abs(self.pos))/2)
            
        elif self.pos[1] == 1:
            if self.pos[0] < self.len:
                self.pos += np.array([a[0],0])
            elif a.tolist() in [[-1,0],[0,-1],[0,0]]:
                self.pos += a
            if self.pos[0] < 0:
                self.pos = np.array([0,0])
                return 1
            
        return -1
    
    def reset(self):
        self.pos = np.array([0,0])
