this is code sample exhibit the ability to write the code with object-oriented language.
the code is written by python, requires python and packages: numpy, matplotlib, scikit-image.
there is a jupyter notebook sample as well. 

the "load_tiff" class loads the diffraction pattern, which is common data in synchrotron or neutron community.
there are also special classes, "exp_param" and "coord_trans", to input the experiment configure and calculate the correlated
coordinate system, which will be utilized in later processing class.
the "data_reduction" includes the class to reduce the 2D diffraction pattern to 1D intensity profile.
the "data_visual" enable the display of 2D and 1D data.

repository includes a sample diffraction pattern, "xs_sample.tiff".

to test the reliability of code could run the script in the command line.

$python test_run.py

the author suggests install Conda environment includes packages mentioned above to run this test code.
under Conda environment, code could be run through jupyter notebook by "test_run.ipynb".

