import numpy as np
from coord_trans import Coord

class ExpConfig(Coord):
    '''
    loading experiment configuration, which is usually provided at beamline during beamrun
    center_x and center_y are beam center position in pixel
    wavelength is beam wavelength, unit is A
    pixel size is correlated to detector pixel size
    detector distance corresponds to sample to detector distance
    
    Calculation of detector coordinate, cartesian and reciprocal coordinate were inheritant 
    from Coord class 
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
                         center_x=center_y,
                         center_y=center_x,
                         wavelength=wavelength,
                         pixel_size=pixel_size,
                         detector_distance=detector_distance,)
        self.reciprocal()
               