import numpy as np

class proc:
    """
    circularlt averaging 2D data to 1D intensity profile
    Parameters
    ------
    im: 2D image, diffraction patter
    exp: experiment configuration and perform correlated coordinate for data reduction
    """
    def __init__(self,im,exp):
        self.im   = np.copy(im).astype(float)
        self.Qmap = exp.Q
    
    def data1d(self,q):
        """
        function process circularly averaging of 
        2D diffraction pattern to 1D intensity profile
        
        Parameters
        ----------
        q: a 1D numpy array determine the interested q range
        
        Returns
        -------
        Iq: 1D intensity profile
        """
        dd = self.im.flatten()
        qd  = self.Qmap.flatten()
        qd = qd[np.isnan(dd)==0]
        dd = dd[np.isnan(dd)==0]
        bins = np.append([2*q[0]-q[1]],q)
        bins += np.append(q,[2*q[-1]-q[-2]])
        bins *= 0.5
        self.Iq,dI = np.histogram(qd,bins=bins,weights=dd)
        return self.Iq
