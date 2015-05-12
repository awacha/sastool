import sastool
import numpy as np
import scipy.optimize

Nx=100

np.random.seed(0)

noise_amplitude=np.random.rand(Nx)

noise=noise_amplitude*np.random.randn(len(noise_amplitude))

x=np.linspace(0,100,Nx)
y=4*x-10+noise
dy=noise_amplitude

#np.savetxt('linear_with_noise.txt',np.vstack((x,y,dy)).T)

data=np.loadtxt('linear_with_noise.txt')[:5,:]
x=data[:,0]
y=data[:,1]
dy=data[:,2]

p,dp,stat=sastool.misc.easylsq.nlsq_fit(x,y,dy,lambda x,a,b:a*x+b,[0,0])

print("A: ",p[0]," +/- ",dp[0])
print("B: ",p[1]," +/- ",dp[1])
print("Chi2: ",stat['Chi2'])
print("R2: ",stat['R2'])
print("Chi2_reduced: ",stat['Chi2_reduced'])

p1,pcov1=scipy.optimize.curve_fit(lambda x,a,b:a*x+b,x,y,[0,0],dy)
print("A1: ",p1[0]," +/- ",pcov1[0][0]**0.5)
print("B1: ",p1[1]," +/- ",pcov1[1][1]**0.5)

print("Covariance:")
print(stat['Covariance'])

print("Correlation:")
print(stat['Correlation_coeffs'])
