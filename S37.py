# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt # for showing image
from pylab import *
from lmfit import Model
import sys
#from lmfit.models import GaussianModel, ExponentialModel, LorentzianModel
from skimage.draw import polygon

plt.close('all')

sys.path.append('/Users/jiliangliu/Downloads/简单体系SAXS data/第五次测试/')
from cir_ave import oneD_intensity

from skimage.io import imread
import os,glob
address = '/Users/jiliangliu/Downloads/简单体系SAXS data/第六次测试/'
os.chdir(address)
list1 = glob.glob('S37.tif')

im = imread(list1[0])
image_x0=715.09#715.02
image_y0=941.08#941.15

'''
mask = np.zeros(np.shape(im))
c = np.array([0,694,706,0])
r = np.array([1606,945,959,1630])
rr,cc = polygon(r,c) 
mask[rr,cc] = 1.
mask[936:947,:] = 1.
#c = np.array([591,668,668,694,695,603,591])
#r = np.array([994,954,948,940,948,1002,994])
#rr,cc = polygon(r,c) 
#mask[rr,cc] = 1.
#c = np.array([724,777,779,727])
#r = np.array([928,901,906,934])
#rr,cc = polygon(r,c) 
#mask[rr,cc] = 1.

mask = mask.astype(bool)
mask[193:212,:] = True;mask[406:423,:] = True;mask[617:636,:] = True;mask[829:848,:] = True;mask[1041:1060,:] =True;mask[1253:1272,:] =True;
mask[1466:1484,:] = True; mask[:,485:495] = True; mask[:,979:989] = True

I = oneD_intensity(im=im,xcenter=image_x0,ycenter=image_y0,mask = mask).cir_ave()
#I = oneD_intensity(im=im,xcenter=image_x0,ycenter=image_y0).cir_ave()
'''

def polar_coord(im,xcenter,ycenter):
    shape_ind = np.asarray(np.shape(im))
    x = np.arange(1,shape_ind[1]+1,1)
    y = np.arange(1,shape_ind[0]+1,1)
    #xx is column index of input image, yy is row index of input image
    xx , yy = np.meshgrid(x, y) 
    # produce the x-coord for matrix with origin at scattering center
    xx = xx - xcenter
    # produce the y-coord for matrix with origin at scattering center
    yy = yy - ycenter
    #produce azimuth coord 
    azimuth = np.arctan(yy/xx)
    azimuth = azimuth  + (xx<0)*np.pi +((xx>=0)&(yy<0))*2*np.pi
    #produce radius coord
    radius = np.sqrt(xx**2+yy**2)
    radius = np.round(radius)
    radius = radius.astype(int)
    return radius,azimuth

radius,azimuth = polar_coord(im,image_x0,image_y0)

mask = ((azimuth<=np.radians(105))&(azimuth>=np.radians(75)))
mask += (azimuth<=np.radians(15))
mask += (azimuth>=np.radians(345))

mask[193:212,:] = False;mask[406:423,:] = False;mask[617:636,:] = False;
mask[829:848,:] = False;mask[1041:1060,:] =False;mask[1253:1272,:] =False;
mask[1466:1484,:] = False; mask[:,485:495] = False; mask[:,979:989] = False;

I = oneD_intensity(im=im,xcenter=image_x0,ycenter=image_y0,mask = ~mask).cir_ave()

wavelength = 1.24
std = 2.03#2.03
pixel_size = 172
q = 2*np.pi*np.sin(np.arcsin(pixel_size*np.arange(0,len(I))*1e-6/std))/wavelength

fig,ax = plt.subplots()
plt.imshow(np.log(im),vmin=6,vmax=10)

    
def sphere( q, r, scale=1.0, contrast=0.1, background=0.0):
     
    V = (4/3)*np.pi*(r**3)
 
    return (scale/V)*((3*V*contrast*(sin(q*r)-q*r*cos(q*r) )/( (q*r)**3 ))**2) + background
    

def Form_Factor(q, r, scale=1.0, contrast=0.1, background=0.0):
    V = (4/3)*np.pi*(r**3)
    return (scale/V)*((3*V*contrast*(sin(q*r)-q*r*cos(q*r) )/(q*r)**3 )) + background

def gaussian(r0,sigmaP):
    r0 = r0*sigmaP
    return exp(-(r0)**2/(2*sigmaP**2))/np.abs(sigmaP)/2**.5/pi**.5
    
def peak_shape(q,mu,sigma):
    
    return (2/pi/sigma)*exp(-4*(q-mu)**2/pi**2/sigma**2) #this is gaussian like shape
    #return (1/pi/np.abs(sigma))*(sigma**2/((q-mu)**2+sigma**2)) #this is lorentz shape

def debye_waller(q,a,sigmaD):
    
    return exp(-sigmaD**2*a**2*q**2)
    
def diffuse_scattering(q,r,r0,sigmaP,scale=1.0):
    sigmaP = np.abs(sigmaP)
    F_AVE_2 = np.zeros((len(q),))
    F_2_AVE = np.zeros((len(q),))
    V = (4/3)*np.pi*(r**3)
    poly_AVE = 0
    for i in range(len(r0)):
        F_AVE_2 = F_AVE_2+Form_Factor(q, r-r0[i]*sigmaP)*gaussian(r0[i],sigmaP)*(r0[1]-r0[0])*sigmaP
        F_2_AVE = F_2_AVE+sphere(q, r-r0[i]*sigmaP)*gaussian(r0[i],sigmaP)*(r0[1]-r0[0])*sigmaP
        poly_AVE = poly_AVE+gaussian(r0[i],sigmaP)*(r0[1]-r0[0])*sigmaP
    #return two var
    #diffuse_scattering[0] is beta(q) is |<F(q)>|^2/<|F(q)|^2>, diffuse_scattering[1] is form factor
    return (scale/V)*(F_AVE_2/poly_AVE)**2/(F_2_AVE/poly_AVE),(F_2_AVE/poly_AVE)
    
def Z0(q,qhkl,Fhkl,sigma):
    Z0 = np.zeros((len(q),))
    for i in range(len(qhkl)):
        Z0 = Z0+Fhkl[i]*peak_shape(q,qhkl[i],np.abs(sigma))
        
    return (1/q**2)*Z0

def structure_factor(q,r,a,qhkl,Fhkl,c,sigmaD,sigma,sigmaP,r0):
    
    return (1+\
            c*Z0(q,qhkl,Fhkl,sigma)*debye_waller(q,a,sigmaD)/\
            diffuse_scattering(q,r,r0,sigmaP)[1]-\
            debye_waller(q,a,sigmaD)*diffuse_scattering(q,r,r0,sigmaP)[0])

def hkl_params(a_inds,c_inds,b1,b2,b3):
    h_inds=np.arange(-8,9,1)
    k_inds=np.arange(-8,9,1)
    l_inds=np.arange(-8,9,1)
    
    
    lattice={}
    d_hkl=np.zeros((10000,))*nan
    #f_hkl=np.zeros((10000,))*nan
    m_hkl=np.zeros((10000,))*nan
    indice = np.zeros((10000,3))*nan
    num_vertex = 0
    tetragonal = True
    #Primitive=True
    for n in range(len(h_inds)):
        for nn in range(len(k_inds)):
            for nnn in range(len(l_inds)):
                lattice[num_vertex]=np.asarray([h_inds[n],k_inds[nn],l_inds[nnn]])
                if tetragonal == True:
                    d_hkl[num_vertex] = ((h_inds[n]**2+k_inds[nn]**2)/a_inds**2+\
                                        l_inds[nnn]**2/c_inds**2)
                    
                    m_hkl[num_vertex] = np.abs(1+\
                                        0*cos(-2*np.pi*(h_inds[n]*(b1)+k_inds[nn]*(b2)\
                                        +l_inds[nnn]*(b3)))+\
                                        0*sin(-2*np.pi*(h_inds[n]*(b1)+k_inds[nn]*(b2)\
                                        +l_inds[nnn]*(b3)))*1j)**2
                                        
                    indice[num_vertex] = np.array([h_inds[n],k_inds[nn],l_inds[nnn]])*1.
                        
                num_vertex = num_vertex+1
                
    r_d,r_d_i=np.unique(np.round(d_hkl[np.isnan(d_hkl)==0],12), return_inverse=True)
    
    multiplies = np.bincount(r_d_i)
    
    multiplies1 = np.zeros((len(multiplies),))
    for _ in range(len(multiplies)):
        multiplies1[_] =  multiplies[_]*\
        m_hkl[np.isnan(d_hkl)==0][np.argmin(np.abs(d_hkl[np.isnan(d_hkl)==0]-r_d[_]))]
    
    a=np.asarray([16.882,18.591,25.558,5.860])
    b=np.asarray([0.461,8.622,1.483,36.396])
    c=12.066
    
    def scattering_factor(a,b,c,theta):
        f = np.zeros((len(theta),))+c
        for i in range(len(theta)):
            for j in range(len(a)):
                f[i]=f[i]+a[j]*np.exp(-b[j]*theta[i]**2)
                
            
        return f
        
    f=scattering_factor(a=a,b=b,c=c,theta=q)
    
    q_hkl=2*pi*(r_d)**.5
    inds = list()
    if tetragonal ==True:
        for j in range(1,len(r_d)):
            inds.append(np.argmin(np.abs(q-q_hkl[j])))
        q_hkl=q_hkl[1:]
    
    Fhkl=np.zeros((len(inds),))        
    for i in range(len(inds)):
        if tetragonal ==True:
            Fhkl[i]=f[inds[i]]*multiplies1[i+1]/q[inds[i]]  ##not sure should divid by q or not here
    return q_hkl,Fhkl
    
def Iq(q,r,a,qhkl,Fhkl,c,sigmaD,sigma,sigmaP,r0,scale):
    a = scale*diffuse_scattering(q,r,r0,sigmaP)[1]*structure_factor(q,r,a,qhkl,Fhkl,c,sigmaD,sigma,sigmaP,r0)#+scale2
    return np.log(a)

b1 = 0.5
b2 = 0.5
b3 = 0.5

a_inds = 582.#848.#827.
c_inds = 582.#1481.#1460.
    
r=50.

q_hkl,Fhkl = hkl_params(a_inds,c_inds,b1,b2,b3)

c= 3.6329688590223275e-2
sigmaD= 0.04
sigma = 1.*(q[1]-q[0])
sigmaP= 0.1
r0=np.asarray([-5,-4,-3,-2,-1,0,1,2,3,4,5])*2.
scale = 1.

form_factor = diffuse_scattering(q,48.,r0,0.001)[1]

weight=np.ones((len(q),))
weight[120:200]=3
weight=weight[60:600]

I_test = I[60:200]
q_test = q[60:200]
gmod = Model(Iq,independent_vars=['q','a','qhkl','Fhkl','r0'])#,weights=weight/np.sum(weight))
pars = gmod.make_params()
pars['r'].set(value=45.2,min=44.,max=46.)
pars['c'].set(value=c,min=0.)
pars['sigma'].set(value=sigma,min = sigma*0.8, max = sigma*1.6)
pars['sigmaD'].set(value=sigmaD,min=0.001,max=0.06)#0.04
pars['sigmaP'].set(value=6.1,min=6.,max= 6.6)
pars['scale'].set(value=2.,min=0.)

result = gmod.fit(np.log(I_test),q=q_test,a=a_inds*1.,qhkl=q_hkl,Fhkl=Fhkl,\
                  r0=r0,c=c,params=pars)#,weights=weight/np.sum(weight))
                                    
r1 = result.params['r'].value  
c1 = result.params['c'].value
sigma1 = result.params['sigma'].value
sigmaD1 = result.params['sigmaD'].value
sigmaP1 = result.params['sigmaP'].value
scale1 = result.params['scale'].value  

intensity = Iq(q,r1,a_inds*1.,q_hkl,Fhkl,c1,sigmaD1*1.,sigma1*1.,sigmaP1,r0,scale1)

structure_factor_test = I/diffuse_scattering(q,r1,r0,sigmaP1)[1]/I[110]*\
                         diffuse_scattering(q,r1,r0,sigmaP1)[1][110]#
          
structure_factor_fit = np.exp(intensity)/diffuse_scattering(q,r1,r0,sigmaP1)[1]/\
                       np.exp(intensity)[110]*\
                       diffuse_scattering(q,r1,r0,sigmaP1)[1][110]

###########################
#下面是plot的commend                 
fig,ax = plt.subplots()
ax.plot(q,np.log(I),q_test,result.best_fit)

fig,ax = plt.subplots()
plt.semilogy(q,I,linewidth=4,label='S37')
plt.semilogy(q,np.exp(intensity),linewidth=3,label='fitting')
plt.axis([0,0.2,10,10000])
plt.xticks([0.,0.1,0.2])
plt.yticks([0.1,1,10,100,1000,50000])
plt.tick_params(labelsize=24,pad = 10)
plt.ylabel(r'$I(q)\,\,(\rm{a.u.})$',fontsize=32)
plt.xlabel(r'$q\,\,(\rm{\AA^{-1}})$',fontsize=32)
plt.legend()
plt.tight_layout()

fig,ax = plt.subplots()
plt.semilogy(q,I,linewidth=4,label='S37')
plt.semilogy(q,np.exp(intensity),linewidth=3,label='fitting')
plt.axis([0,0.08,100,50000])
plt.xticks([0.,0.02,0.04,0.06,0.08])
plt.yticks([100,1000,10000])
plt.tick_params(labelsize=24,pad = 10)
plt.ylabel(r'$I(q)\,\,(\rm{a.u.})$',fontsize=32)
plt.xlabel(r'$q\,\,(\rm{\AA^{-1}})$',fontsize=32)
plt.legend()
plt.tight_layout()

fig,ax = plt.subplots()
plot(q,structure_factor_test,linewidth=4,label='S37')
plot(q,(structure_factor_fit-1)*1.+1,linewidth=3,label='fitting');
plt.axis([0,0.1,0,5])
plt.xticks([0.,0.02,0.04,0.06,0.08])
plt.yticks([0,1,2,3,4,5])
plt.tick_params(labelsize=24,pad = 10)
plt.ylabel(r'$S(q)\,\,(\rm{a.u.})$',fontsize=32)
plt.xlabel(r'$q\,\,(\rm{\AA^{-1}})$',fontsize=32)
plt.legend()
plt.tight_layout()

plt.show()

q1 = q.reshape(len(q),1);
I1 = I.reshape(len(I),1);
I2 = np.exp(intensity.reshape(len(intensity),1))
F  = diffuse_scattering(q,r1,r0,sigmaP1)[1]*I[45]/diffuse_scattering(q,r1,r0,sigmaP1)[1][45]
F1 = F.reshape(len(F),1)
S1 = structure_factor_test.reshape(len(structure_factor_test),1)
S2 = structure_factor_fit.reshape(len(structure_factor_fit),1)

txtname = list1[0].split('.')[0]+'.txt'
np.savetxt(txtname,np.concatenate((q1,I1,I2,F1,S1,S2),axis=1),fmt=('%10.6f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f'),\
           delimiter=',',
           header=('%8s,%10s,%10s,%10s,%10s,%10s'%('q','I(q)','I_fit','P(q)','S(q)','S_fit(q)')),
           newline='\n')
