import numpy as np
import matplotlib.pyplot as plt
from exp_weibull import phase_pdf
from exp_weibull import phase_hist

Omega = 1.0
alpha = 1.0
K = 1000000
t1 = np.linspace(0, 2 * np.pi / alpha, 1000)

graph = plt.gca(projection='polar')
graph.axes.get_yaxis().set_ticks([])

for m in [1.0, 1.25, 1.5, 1.8, 2.3, 3.0, 4.0]:
    p = phase_pdf(t1, alpha, m, Omega)
    plt.plot(t1, p, 'k-')

for m in [1.0, 3.0, 4.0, 5.0]:
    x, y = phase_hist(alpha, m, Omega, K, 50)
    plt.plot(x, y, 'ko', mfc='none')

p = phase_pdf(t1,alpha,5.0,Omega)
plt.plot(t1,p,'k-',label=r'$\alpha=1,~\mu \geq 1$')

alpha = 2.0
t1 = np.linspace((np.pi/2000),(1-np.pi/2000)*np.pi/alpha,1000)
t2 = np.linspace(np.pi/alpha,np.pi,1000)
t3 = np.linspace(np.pi,3*np.pi/2,1000)
t4 = np.linspace(3*np.pi/2,2*np.pi,1000)

tset = [t1, t2, t3, t4]

muset = [0.7, 0.8, 0.9]
for m in muset:
    p = phase_pdf(t1,alpha,m,Omega)
    for t in tset:
        plt.plot(t,p,'k-',color='.75')

p = phase_pdf(t1,alpha,0.95,Omega)
plt.plot(t1,p,'k-',color='.75')
plt.plot(t2,p,'k-',color='.75')
plt.plot(t3,p,'k-',color='.75')
plt.plot(t4,p,'k-',color='.75',label=r'$\alpha=2,~\mu < 1$')

plt.rc('text',usetex=True)
plt.rc('font',**{'family':'serif','serif':['Times'],'size':16})
plt.ylim([0,0.25])
plt.legend(shadow=True,fontsize=14,loc=(0.55,0.925))
plt.show()
