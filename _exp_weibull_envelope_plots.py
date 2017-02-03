import numpy as np
import matplotlib.pyplot as plt
from exp_weibull import envelope_pdf
from exp_weibull import env_pdf
from exp_weibull import envelope_hist

r = np.linspace(5e-2,5,1000)
K = 1000000

pdf1 = envelope_pdf(r,.8,.5,1.0)
pdf2 = env_pdf(r,.8,.5,1.0)
plt.plot(r,pdf2,'k-',label=r'$p_R(r) = \int_{0}^{4\pi/\alpha}p_{R,\Theta}(r,\theta)\;\mathrm{d}\theta$')
plt.plot(r,pdf1,'k--',label=r'[5, eq. (15)]')

pdf1 = envelope_pdf(r,1.0,.75,1.0)
pdf2 = env_pdf(r,1.0,.75,1.0)
plt.plot(r,pdf2,'k-',r,pdf1,'k--')

pdf1 = envelope_pdf(r,1.2,1.0,1.0)
pdf2 = env_pdf(r,1.2,1.0,1.0)
plt.plot(r,pdf2,'k-',r,pdf1,'k--')
x,y = envelope_hist(1.2,1.0,1.0,K,50)
plt.plot(x,y,'ko',mfc='none')

pdf1 = envelope_pdf(r,1.5,1.5,1.0)
pdf2 = env_pdf(r,1.5,1.5,1.0)
plt.plot(r,pdf2,'k-',r,pdf1,'k--')

pdf1 = envelope_pdf(r,2.5,2.0,1.0)
pdf2 = env_pdf(r,2.5,2.0,1.0)
plt.plot(r,pdf2,'k-',r,pdf1,'k--')
x,y = envelope_hist(2.5,2.0,1.0,K,50)
plt.plot(x,y,'ko',mfc='none')

pdf1 = envelope_pdf(r,4.0,2.0,1.0)
pdf2 = env_pdf(r,4.0,2.0,1.0)
plt.plot(r,pdf2,'k-',r,pdf1,'k--')
x,y = envelope_hist(4.0,2.0,1.0,K,50)
plt.plot(x,y,'ko',mfc='none')

plt.rc('text',usetex=True)
plt.rc('font',**{'family':'serif','serif':['Times'],'size':16})
plt.ylabel(r"$p_R(r)$",fontname='TimesNewRomanPS-BoldMT',fontsize=16)
plt.xlabel(r"$r$",fontname='TimesNewRomanPS-BoldMT',fontsize=16)
plt.ylim([0,2.0])
plt.xlim([5e-2,3.0])
ax = plt.axes()
ax.arrow(1, 0.1, 0.4, 1.2, head_width=0.04, head_length=0.05, fc='k', ec='k')
plt.text(1.5,1.4,r'$(\alpha,\mu) = \{(0.8, 0.5), (1.0, 0.75),$',fontsize=16)
plt.text(1.96,1.29,r'$(1.2, 1.0), (1.5,1.5),$',fontsize=16)
plt.text(1.96,1.18,r'$(2.5, 2.0), (4.0, 2.0)\}$',fontsize=16)
plt.legend(shadow=True,loc=1,numpoints=1,fontsize=14)
plt.grid()
plt.show()
