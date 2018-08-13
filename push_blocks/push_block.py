import numpy as np

class block:
    def __init__(self,loc):
        self.loc = loc
        
        
class target:
    def __init__(self,loc):
        self.loc = loc
        self.occ = False

class block_arena:
    def __init__(self,size,ntarg,nblock):
        self.size = size
        self.ntarg = ntarg
        self.nblock = nblock
        self.arena = np.array([[i,j] for i in range(0,size) for j in range(0,size)])

        temp = np.random.choice(range(len(self.arena)),size = ntarg + nblock + 1,replace = False)
        temp = self.arena[temp]
        self.targets = [target(l) for l in temp[:ntarg]]
        self.blocks = [block(l) for l in temp[ntarg:-1]]
        self.arena = [a for a in self.arena]
        
        self.loc = temp[-1]
        
    def block_occ(self,loc):        
        for k in self.blocks:
            if np.array_equal(loc,k.loc):
                return True
        return False

    def targ_occ(self,loc):        
        for k in self.targets:
            if np.array_equal(loc,k.loc):
                return True
        return False
    
    def in_arena(self,loc):
        for k in self.arena:
            if np.array_equal(loc,k):
                return True 
        return False
    
    def isopen(self,loc):
        if (in_arena(loc) and not block_occ(loc)):
            return True
        else:
            return False
        
    def get_arena(self):
        temp = np.zeros([3,self.size,self.size])
        for k in self.targets:
            temp[0,k.loc[0],k.loc[1]] = 1
        for k in self.blocks:
            temp[1,k.loc[0],k.loc[1]] = 1
            
        temp[2,self.loc[0],self.loc[1]] = 1
        
        return temp
   
    def get_objects(self,loc):
        out = {"blocks":[],"target":[]}
        
        for k in self.blocks:
            if np.array_equal(k.loc,loc):
                out["blocks"].append(k)
        for k in self.targets:
            if np.array_equal(k.loc,loc):
                out["blocks"].append(k)
        return out

    def move(self,DIR):
        dir = np.array(DIR)
        
        assert dir.tolist() in ([0,1],[1,0],[0,-1],[-1,0])
            
        if self.in_arena(self.loc + dir) == False:
            return 0

        assert self.in_arena(self.loc + dir) == True
        
        vals = self.get_objects(self.loc + dir)
        
        if self.block_occ(self.loc + dir):
            assert len(vals["blocks"]) == 1
            if self.block_occ(self.loc + 2*dir)==False:
                if self.in_arena(self.loc + 2*dir) == True:
                    vals["blocks"][0].loc += dir
                    self.loc += dir
                    return 1
        else:
            self.loc = self.loc + dir
            return 1
        
    def target_full(self):
        for k in self.targets:
            if not self.block_occ(k.loc):
                return False
        return True