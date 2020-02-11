=====
intro
=====

this is code sample for exhibiting the coding ability

.. code-block:: python

   import code_sample

Functions
---------
.. autofunction:: im_load.load_tiff

Example
--------
.. ipython:: python
   
   from im_load import load_tiff
   import numpy as np
   import os,glob
   import warnings
   warnings.simplefilter("ignore")
   path = os.path.abspath('../../code_sample/code')
   os.chdir(path)
   fn = glob.glob('*.tif')[0]
   img = load_tiff(fn).im
   
   from exp_param import ExpConfig
   dexp = ExpConfig(im_x = img.shape[1],
                    im_y = img.shape[0],
                    center_x = 715.09,
                    center_y = 914.08,
                    wavelength = 1.24,
                    pixel_size = 172,
                    detector_distance = 2.03,
                    )
   
   from data_reduction import proc
   d = proc(img,dexp)
   q = np.linspace(0.001,0.2,300)
   d1d = d.data1d(q)
   
   from data_visual import data_show
   show = data_show(d,dexp)
   show.show2D(vmax = 10000)
   show.show1D(q)

Plot
-------
.. plot::
   
   from im_load import load_tiff
   import numpy as np
   import os,glob
   import warnings
   warnings.simplefilter("ignore")
   path = os.path.abspath('../../code')
   os.chdir(path)
   fn = glob.glob('*.tif')[0]
   img = load_tiff(fn).im
   
   from exp_param import ExpConfig
   dexp = ExpConfig(im_x = img.shape[1],
                    im_y = img.shape[0],
                    center_x = 715.09,
                    center_y = 914.08,
                    wavelength = 1.24,
                    pixel_size = 172,
                    detector_distance = 2.03,
                    )
   
   from data_reduction import proc
   d = proc(img,dexp)
   q = np.linspace(0.001,0.2,300)
   d1d = d.data1d(q)
   
   from data_visual import data_show
   show = data_show(d,dexp)
   show.show2D(vmax = 10000)
   show.show1D(q)
   plt.show()
