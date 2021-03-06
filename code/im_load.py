import numpy as np
from skimage.io import imread

class load_tiff:
    '''
    load tiff image
    
    Parameters
    ----------
    fn: file name
    path: directory include file
    '''
    def __init__(self,fn,path=None):
        if path:
            self.fn = path+fn
        else:
            self.fn = fn        
        self.im = imread(fn)
