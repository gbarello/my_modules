import numpy

class long_walker:

    def __init__(self,length):

        self.len = length

        self.pos = np.array([0,0])

    def update(a):
        assert a.tolist() in [[0,0],[1,0],[-1,0],[0,1],[0,-1]]

        if self.pos[1] == 0:
            if self.pos[0] < self.len:
                self.pos += np.array([a[0],0])
            elif a.tolist() in [[-1,0],[0,1],[0,0]]:
                self.pos += a
            
            self.pos = max(self.pos,0)
            
        elif self.pos[1] == 1:
            if self.pos[0] < self.len:
                self.pos += np.array([a[0],0])
            elif a.tolist() in [[-1,0],[0,-1],[0,0]]:
                self.pos += a
            if self.pos[0] < 0:
                self.pos = np.array([0,0])
                return 1
            
        return 0
