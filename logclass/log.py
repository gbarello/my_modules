import utilities as util
import numpy as np
class log:
    def __init__(self,entries = [],fname = "",PRINT = False):

        if fname != "":
            self.data = np.loadtxt(fname)

        else:
            self.data = [entries]
            if PRINT:
                self.print_list(entries)
                
    def print_list(self,dat):
        t = str(dat[0])
        for e in dat[1:]:
            t += "\t{}".format(e)
        print(t)

    def log(self,data,PRINT = True):

        if PRINT:
            self.print_list(data)

        self.data.append(data)
    def save(self,fname):
        np.savetxt(fname,self.data[1:])
