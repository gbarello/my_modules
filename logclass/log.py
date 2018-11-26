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

class running_log:
    def __init__(self,fname,header="",PRINT = False):

        self.fname = fname

        if type(header) != list:
            pr = header
            F = open(fname,"w")
            F.write(pr)
            F.close()
        else:
            pr = "{}".format(header[0])
            for t in header[1:]:
                pr += ",{}".format(t)
                
            F = open(self.fname,"w")
            F.write(pr + "\n")
            F.close()
            
        
        if PRINT:
            print(pr)
        
    def log(self,data,PRINT = True):

        pr = "{}".format(data[0])
        for t in data[1:]:
            pr += ",{}".format(t)

        F = open(self.fname,"a")
        F.write(pr + "\n")
        F.close()

        if PRINT:
            print(pr)
