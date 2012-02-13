import os

fsns=[3278,3564,8280,3279,7000,8283]

for f in fsns:
    fo=open('submit_%d.sh'%f,'wt')
    fo.write('#!/bin/sh\ncd /home/andris/sastool/tests/centering\npython beamfinding_tests.py %d\n'%f)
    fo.close()
    os.system('sbatch -p plutoonly submit_%d.sh'%f)

