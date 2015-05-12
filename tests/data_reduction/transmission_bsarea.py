'''
Created on Apr 4, 2012

@author: andris
'''

import sastool
import os
import matplotlib.pyplot as plt
import numpy as np

#datarootdir='/home/andris/kutatas/desy/2011/1216Bota'
datarootdir='/home/andris/kutatas/desy/2012/0324Bota'

subdirs=['.','data1','cbf_files']

pri = [459,474,486,493];


def get_I0(data,header,pri):
    bsarea=data[pri[2]:pri[3],pri[0]:pri[1]];
    return bsarea.sum()/header['Monitor'],np.sqrt(header['MonitorError']**2*bsarea.sum()**2/header['Monitor']**4+bsarea.sum()/header['Monitor']**2)

datadirs=[os.path.join(datarootdir,sd) for sd in subdirs];

data,header=sastool.io.b1.read2dB1data(list(range(10)),'org_%05d',dirs=datadirs)

I0=[get_I0(d,h,pri) for d,h in zip(data,header)]

for i in range(len(data)):
    print(header[i]['FSN'],header[i]['Title'],header[i]['Transm'], I0[i][0]/I0[0][0], np.sqrt(I0[i][0]**2*I0[0][1]**2/I0[0][0]**4+I0[i][1]**2/I0[0][0]));
