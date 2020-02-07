import numpy as np

class proc:
    '''
    circularlt averaging 2D data to 1D intensity profile
    '''
    def __init__(self,im,exp):
        '''
        im is collected diffraction patter
        exp is a class including experiment configuration 
        and correlated coordinate for data reduction
        '''
        self.im   = np.copy(im).astype(float)
        self.Qmap = exp.Q
    
    def data1d(self,q):
        '''
        q is a numpy array determine the interested q range
        '''
        dd = self.im.flatten()
        qd  = self.Qmap.flatten()
        bins = np.append([2*q[0]-q[1]],q)
        bins += np.append(q,[2*q[-1]-q[-2]])
        bins *= 0.5
        self.Iq,dI = np.histogram(qd,bins=bins,weights=dd)
        return self.Iq