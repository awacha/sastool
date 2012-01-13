import sastool
import numpy as np
import scipy.io

xmin=110
xmax=130
ymin=110
ymax=130

data,header=sastool.io.b1.read2dB1data(7000, 'ORG%05d', '.')
mask=scipy.io.loadmat('mask.mat')['mask']

distslice=np.ones_like(data[0])*np.nan
distazimfold=np.ones_like(data[0])*np.nan
bcxazimfold=np.ones_like(data[0])*np.nan
bcyazimfold=np.ones_like(data[0])*np.nan
bcxslice=np.ones_like(data[0])*np.nan
bcyslice=np.ones_like(data[0])*np.nan
for x in range(xmin,xmax+1):
    for y in range(ymin,ymax+1):
        print "X: ",x,"    Y: ",y
        orig_initial=[x+1,y+1]
        orig1=sastool.utils2d.centering.findbeam_azimuthal_fold(data[0], orig_initial, mask.astype(np.uint8), dmin=40, dmax=90)
        orig2=sastool.utils2d.centering.findbeam_slices(data[0], orig_initial, mask.astype(np.uint8),dmin=40, dmax=90)
        distazimfold[x,y]=np.sqrt((orig_initial[0]-orig1[0])**2+(orig_initial[1]-orig1[1])**2)
        distslice[x,y]=np.sqrt((orig_initial[0]-orig2[0])**2+(orig_initial[1]-orig2[1])**2)
        bcxazimfold[x,y]=orig1[0]
        bcyazimfold[x,y]=orig1[1]
        bcxslice[x,y]=orig2[0]
        bcyslice[x,y]=orig2[1]
np.save('distazimfold.npy',distazimfold)
np.save('distslice.npy',distslice)
np.save('bcxazimfold.npy',bcxazimfold)
np.save('bcyazimfold.npy',bcyazimfold)
np.save('bcxslice.npy',bcxslice)
np.save('bcyslice.npy',bcyslice)
