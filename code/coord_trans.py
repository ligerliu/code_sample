import numpy as np

class Coord:
    '''
    transform detector coordinate system 
    to sample coordinate system then further transform
    to reciprocal coordinate for data reduction
    '''
    def __init__(self,
                 im_x=100,
                 im_y=100,
                 center_x=0,
                 center_y=0,
                 wavelength=1,
                 pixel_size=172,
                 detector_distance=1,
                ):
        self.X_size = im_x
        self.Y_size = im_y
        self.bm_centrx  = center_x
        self.bm_centry  = center_y
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.Dd         = detector_distance
        self.Y,self.X = np.meshgrid(np.arange(self.X_size)+0.5,
                                    np.arange(self.Y_size)+0.5)
        self.X -= self.bm_centrx
        self.Y -= self.bm_centry
        self.Qx = 2*np.pi*np.arcsin(self.X*self.pixel_size*1e-6/self.Dd)/self.wavelength
        self.Qy = 2*np.pi*np.arcsin(self.Y*self.pixel_size*1e-6/self.Dd)/self.wavelength
    
    def real(self):
        #R is distance to the image center in pixel
        self.R = np.sqrt(self.X**2+self.Y**2)
    
    def reciprocal(self):
        #Q is distance to the image center in reciprocal space
        self.Q = np.sqrt(self.Qx**2+self.Qy**2)