import numpy as np
import scipy.special as sps
import scipy.integrate as integrate

def envelope_pdf(r,alpha,mu,Omega):
    return (mu*alpha/Omega)*np.power(r/Omega,alpha-1.0)*np.exp(-np.power(r/Omega,alpha))*np.power(1.0-np.exp(-np.power(r/Omega,alpha)),mu-1.0)

def joint_envelope_phase_pdf(r,t,alpha,mu,Omega):
    return alpha*mu*alpha*mu*np.power(r/Omega,alpha-1.0)*np.exp(-np.power(r/Omega,alpha))*np.power(sps.erf(np.power(r/Omega,alpha/2.0)*np.abs(np.cos(alpha*t/2.0)))*sps.erf(np.power(r/Omega,alpha/2.0)*np.abs(np.sin(alpha*t/2.0))),mu-1.0)/(4*np.pi*Omega)

def phase_pdf(t,alpha,mu,Omega):
    p_pdf = np.zeros(len(t))
    for j in range(len(t)):
        jpdf = lambda r: joint_envelope_phase_pdf(r,t[j],alpha,mu,Omega)
        p_pdf[j] = integrate.quad(jpdf,0.0,np.inf,epsrel=1e-9,epsabs=1e-9)[0]
    p_pdf[p_pdf == float('+inf')] = 0.0
    p_pdf[np.isnan(p_pdf)] = 0.0
    return p_pdf

def env_pdf(r,alpha,mu,Omega):
    env_pdf = np.zeros(len(r))
    for j in range(len(r)):
        jpdf = lambda t: joint_envelope_phase_pdf(r[j],t,alpha,mu,Omega)
        env_pdf[j] = integrate.quad(jpdf,0.0,4*np.pi/alpha,epsrel=1e-9,epsabs=0)[0]

    return env_pdf

def rvs(alpha, mu, Omega,K):
    X1 = np.sqrt(np.amax((np.power(Omega,alpha)/2.0)*np.power(np.random.normal(size=(K,mu)),2),1))
    Y1 = np.sqrt(np.amax((np.power(Omega,alpha)/2.0)*np.power(np.random.normal(size=(K,mu)),2),1))

    X2 = -np.sqrt(np.amax((np.power(Omega,alpha)/2.0)*np.power(np.random.normal(size=(K,mu)),2),1))
    Y2 = np.sqrt(np.amax((np.power(Omega,alpha)/2.0)*np.power(np.random.normal(size=(K,mu)),2),1))

    X3 = -np.sqrt(np.amax((np.power(Omega,alpha)/2.0)*np.power(np.random.normal(size=(K,mu)),2),1))
    Y3 = -np.sqrt(np.amax((np.power(Omega,alpha)/2.0)*np.power(np.random.normal(size=(K,mu)),2),1))

    X4 = np.sqrt(np.amax((np.power(Omega,alpha)/2.0)*np.power(np.random.normal(size=(K,mu)),2),1))
    Y4 = -np.sqrt(np.amax((np.power(Omega,alpha)/2.0)*np.power(np.random.normal(size=(K,mu)),2),1))

    return np.array([X1,Y1,X2,Y2,X3,Y3,X4,Y4])

def phase_rvs(alpha, mu, Omega, K):
    Z = rvs(alpha,mu,Omega,K)
    t1 = 2*np.arctan2(Z[1],Z[0])/alpha
    t2 = 2*np.arctan2(Z[3],Z[2])/alpha
    t3 = 2*np.arctan2(Z[5],Z[4])/alpha + 4*np.pi/alpha
    t4 = 2*np.arctan2(Z[7],Z[6])/alpha + 4*np.pi/alpha

    return np.reshape(np.vstack((t1,t2,t3,t4)),(4*K,))

def phase_hist(alpha,mu,Omega,K,Nbins):
    rvs = phase_rvs(alpha,mu,Omega,K)
    y,x_edge = np.histogram(rvs,bins=Nbins,density=True)
    x = 0.5*(x_edge[1:]+x_edge[:-1])
    return np.array([x,y])

def envelope_rvs(alpha,mu,Omega,K):
    Z = rvs(alpha,mu,Omega,K)
    R = np.power(Z[1]*Z[1] + Z[2]*Z[2],1.0/alpha) 
    return R

def envelope_hist(alpha,mu,Omega,K,Nbins):
    rvs = envelope_rvs(alpha,mu,Omega,K)
    y,x_edge = np.histogram(rvs,bins=Nbins,density=True)
    x = 0.5*(x_edge[1:]+x_edge[:-1])
    return np.array([x,y])

