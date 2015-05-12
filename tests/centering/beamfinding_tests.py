#!/usr/bin/python
import sastool
import numpy as np
import scipy.io
import sys
import time

xmin=40
xmax=210
ymin=40
ymax=210

fsn=int(sys.argv[1])

modes=['slice','azim','azimfold']
funcs={'slice':lambda data,orig_initial,mask:sastool.utils2d.centering.findbeam_slices(data[0], orig_initial, mask.astype(np.uint8),dmin=40, dmax=90),
       'azim':lambda data,orig_initial,mask:sastool.utils2d.centering.findbeam_azimuthal(data[0], orig_initial, mask.astype(np.uint8), dmin=40, dmax=90),
       'azimfold':lambda data,orig_initial,mask:sastool.utils2d.centering.findbeam_azimuthal_fold(data[0], orig_initial, mask.astype(np.uint8), dmin=40, dmax=90),
       }
mask=scipy.io.loadmat('mask.mat')['mask']

print("Loading file %d"%fsn)
data,header=sastool.io.b1.read2dB1data(fsn, 'ORG%05d', '.')

dist={}; bcx={}; bcy={}; xtime={}
for i in modes:
    dist[i]=np.ones((256,256),np.double)*np.nan;
    bcx[i]=np.ones((256,256),np.double)*np.nan;
    bcy[i]=np.ones((256,256),np.double)*np.nan;
    xtime[i]=np.ones((256,256),np.double)*np.nan;
for x in range(xmin,xmax+1):
    for y in range(ymin,ymax+1):
        print("X: ",x,"    Y: ",y)
        orig_initial=[x+1,y+1]
        for m in modes:
            orig=[np.nan,np.nan]
            try:
                t0=time.time()
                orig=funcs[m](data,orig_initial,mask)
                t1=time.time()
            except Exception as e:
                print(str(e))
                continue
            dist[m][x,y]=np.sqrt((orig_initial[0]-orig[0])**2+(orig_initial[1]-orig[1])**2)
            bcx[m][x,y]=orig[0]
            bcy[m][x,y]=orig[1]
            xtime[m][x,y]=t1-t0;

for m in modes:
    np.save('dist%s_%d.npy'%(m,fsn),dist[m])
    np.save('bcx%s_%d.npy'%(m,fsn),bcx[m])
    np.save('bcy%s_%d.npy'%(m,fsn),bcy[m])
    np.save('time%s_%d.npy'%(m,fsn),xtime[m])