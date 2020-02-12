import numpy as np
import matplotlib.pyplot as plt

class data_show:
    '''
    input process and experiment configure class
    display the 2D raw data and process 1D intensity
    
    Parameters
    ----------
    d: object, data proc class
    exp: object, exp config class
    '''
    def __init__(self,d,exp):
        self.d   = d
        self.exp = exp
        
    def show2D(self,log=True,vmin=1,vmax=1000):
        """
        correlated reciprocal coordinates were shown
        log scale and value limits were applied as well
        
        Parameters
        ----------
        log: bool, plot with log scale
        vmin: min intensity threshold for plot
        vamx: max intensity threshold for plot
        
        Returns 
        --------
        2D plot with correlated Q coordinate
        """
        if log:
            self.im = self.d.im
            self.im[self.im<=0] = np.nan
            self.im = np.log(self.im)
            self.vmin = np.log(vmin)
            self.vmax = np.log(vmax)
        else:
            self.im = self.d.im
            self.vmin = vmin
            self.vmax = vmax
        Qxmin = np.min(self.exp.Qx)
        Qxmax = np.max(self.exp.Qx)
        Qymin = np.min(-self.exp.Qy)
        Qymax = np.max(-self.exp.Qy)
        plt.subplots()
        plt.imshow(self.im,vmin=self.vmin,vmax=self.vmax,cmap='jet',
                   extent=(Qxmin,Qxmax,Qymin,Qymax))
        plt.xlabel(r'$Q_{x}\,\,(\AA)$')
        plt.ylabel(r'$Q_{y}\,\,(\AA)$')
        plt.axis('image')
        plt.tight_layout()
        
    def show1D(self,q,log=True):
        """
        processed 1D intensity were presented here
        
        Parameters
        ----------
        q: 1D array correlates to the intereting range of q for intensity reduction
        
        Returns
        -------
        plot of 1D intensity profile vs q
        """
        self.I = self.d.data1d(q)
        plt.subplots()
        if log:
            plt.semilogy(q,self.I)
        else:
            plt.plot(q,self.I)
        plt.xlabel(r'$Q_{x}\,\,(\AA)$')
        plt.ylabel(r'$I(Q)$')
            
        
