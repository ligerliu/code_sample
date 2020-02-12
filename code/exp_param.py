import numpy as np
from coord_trans import Coord

class ExpConfig(Coord):
    '''
    loading experiment configuration, which is usually provided at beamline during beamrun
    
    Calculation of detector coordinate, cartesian and reciprocal coordinate were inheritant 
    from Coord class 
   
    Parameters
    ----------
    im_x: column size of input image
    im_y: row size of input image
    center_x: beam center position in column axis 
    center_y: beam center position in row axis
    wavelength: beam wavelength, unit is A
    pixel size: detector pixel size, unit is micron
    detector distance: sample to detector distance
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
        super().__init__(im_x=im_x,
                         im_y=im_y,
                         center_x=center_x,
                         center_y=center_y,
                         wavelength=wavelength,
                         pixel_size=pixel_size,
                         detector_distance=detector_distance,)
        self.reciprocal()
               
