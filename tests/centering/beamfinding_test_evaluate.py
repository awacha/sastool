import os
import numpy as np
import matplotlib.pyplot as plt
import re
import sastool
import matplotlib

matplotlib.rcParams['font.size']=8
xmin=100
xmax=150
ymin=100
ymax=150

modes=['slice','azim','azimfold']

bcxfiles=[f for f in os.listdir('.') if re.match('bcx[a-z]+_([0-9]+).npy',f)]
print bcxfiles
fsns=set([re.match('bcx[a-z]+_([0-9]+).npy',f).group(1) for f in bcxfiles])
print fsns

for f in fsns:
    f=int(f)
    print f
    data,header=sastool.io.b1.read2dB1data(f,'ORG%05d','.')
    plt.clf()
    plt.subplot(3,4,1)
    plt.imshow(data[0],interpolation='nearest')
    modeidx=0
    for m,i in zip(modes,range(len(modes))):
        bcx=np.load('bcx%s_%d.npy'%(m,f))
        bcy=np.load('bcy%s_%d.npy'%(m,f))
        dist=np.load('dist%s_%d.npy'%(m,f))
        plt.subplot(3,4,i*4+2)
        plt.imshow(bcx-np.mean(bcx[np.isfinite(bcx)]),interpolation='nearest')
        plt.axis((xmin-2,xmax+2,ymin-2,ymax+2))
        plt.colorbar()
        plt.subplot(3,4,i*4+3)
        plt.imshow(bcy-np.mean(bcy[np.isfinite(bcy)]),interpolation='nearest')
        plt.axis((xmin-2,xmax+2,ymin-2,ymax+2))
        plt.colorbar()
        plt.subplot(3,4,i*4+4)
        plt.imshow(dist,interpolation='nearest')
        plt.axis((xmin-2,xmax+2,ymin-2,ymax+2))
        plt.colorbar()
    print "Saving image"
    plt.savefig('bftest_%d.pdf'%f)
    