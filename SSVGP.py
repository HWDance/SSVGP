
# Requirements for algorithms
import numpy as np
import scipy.special as sp
from functools import partial
from scipy.spatial.distance import cdist
from sklearn import neighbors

# Requirements for simulations/monitoring algorithms
import time
from IPython.display import clear_output
import inspect
from sklearn.metrics import roc_auc_score

# Requirements For plots and diagnostics
import matplotlib.pyplot as plt
import scipy.stats as sps
from scipy.stats import norm


class kernel_funcs:
    
    # Gaussian kernel
    def gaussian(d,s):
        return s*np.exp(-0.5*d**2)
    
    # Cauchy kernel
    def cauchy(d,s):
        return s*1/(1+0.5*d**2)
        
    # Matern kernel
    def matern0(d,s):
        return s*np.exp(-d) 
    
    # Wave kernel
    def periodic(d,s):
        return np.exp(-2*np.sin(d*np.pi)**2)
    
    # Gaussian kernel gradient
    def grad_gaussian(K,X,l,s):
        Xd=(X-X.T)**2
        return -(K*Xd)*l 
    
    # Cauchy kernel gradient
    def grad_cauchy(K,X,l,s):
        Xd=(X-X.T)**2
        return -(K**2*Xd)*l
    
    def grad_lin(K,X,l,s):
        return 2*l*s*X @ X.T

    # Gradient of K wrt scale s
    def gradK_s(K,s):
        return 2*K/s

    def ARD(l,s,X,kern):
        if kern!="lin":
            Z = X*l
            return kern(cdist(Z,Z),s**2)
        else:
            Z = np.diag(l) @ X.T
            return (Z.T @ Z)*s**2

    def ARDtest(l,s,X,Xtest,kern):
        if kern!="lin":
            return kern(cdist(X*l,Xtest*l),s**2).T
        else:
            Z = np.diag(l) @ X.T
            Ztest = np.diag(l) @ Xtest.T
            return (Ztest.T @ Z)*s**2

class model_funcs:

    # Gradient of marginal likelihood trace term interior (Kinvy @ Kinvy.T - Kinv)
    def grad_logL_A(y,Ktild):
        Kinv = np.linalg.inv(Ktild)
        Kinvy=Kinv @ y
        return  Kinvy @ Kinvy.T - Kinv
    
    # Marginal likelihood
    def logL(y,K):
        L = np.linalg.cholesky(K)
        logL=-1/2*y.T @ np.linalg.solve(K,y)-np.sum(np.log(np.diag(L)))
        return logL
    
class draw_simulation:
    
    def toy_example(n=300,p=100,q=5,noise=0.05):
        a = np.linspace(1,0.5,q)
        X = np.random.multivariate_normal(np.zeros(p), np.eye(p), n+100)
        f = np.sin(a*X[:,:q]).sum(1)
        noise_var = noise*np.var(f)
        Y = (f + np.random.normal(0,noise_var**0.5,n+100)).reshape(n+100,1)
        
        return Y,X,f

 
    def experiment1(n=100,p=1000):

        coefs_inner = np.concatenate((np.repeat(1,4), [3],[5]))
        X = np.random.random((n+20)*p).reshape(n+20,p)
        Z = X[:,:6] * coefs_inner
        F = (np.column_stack((Z[:,:4],(np.sin(Z[:,4:])).reshape(120,2)))).sum(1)
        e = np.random.normal(0,0.05,n+20).reshape(n+20,)
        Y = F+e
        
        return Y,X,F
    
    def experiment2(n, ntest, p, start, end, corrxz,corrzz, r2,block_corr=False,lin = False):
        """
        Function to draw latent function from 2d sinusoidal function - DRAWS DATA RANDOMLY

        n = # samples to train on (may change to ensure grid dimensions work)
        ntest = # samples to test on (may change to ensure grid dimensions work)
        p = # noise variables
        start = linspace start for generator variables
        end = linspace end for generator variables
        corrxz = correlation between noise variables and generator variables
        r2 = r-squared (noise vs. signal)
        lin = True/False for whether X2 is linear generating dimension
        """

        # Drawing generating dimensions (2d)
        xtrain = np.random.random(n)
        ytrain = np.random.random(n)
        xtest = np.linspace(start, end, int(np.sqrt(ntest)))
        ytest = np.linspace(start, end, int(np.sqrt(ntest)))
        ntrain = n 
        ntest = len(ytest)**2
        x1,x2= xtrain,ytrain

        # Creating latent and observed response
        if lin:
            z = 2*x2+np.sin(10*np.pi*x1)
        else:
            z =  +(np.tan(x1)+np.tan(x2)+np.sin(2*np.pi*x1)+np.sin(2*np.pi*x2)
              +np.cos(4*np.pi**2*x1*x2))

        x1test,x2test= np.meshgrid(xtest,ytest)
        if lin:
            ztest = 2*x2test+np.sin(10*np.pi*x1test)
        else:
            ztest =  +(np.tan(x1test)+np.tan(x2test)+np.sin(2*np.pi*x1test)+np.sin(2*np.pi*x2test)
              +np.cos(4*np.pi**2*x1test*x2test))

        F = np.row_stack((z.reshape(ntrain,1),ztest.reshape(ntest,1)))
        sigma = np.sqrt((1-r2)/r2*np.var(F))
        e = np.random.normal(0,sigma, ntrain+ntest).reshape(ntrain+ntest,1)
        Y = F+e
        
        # Plotting training and test surface      
        z_min, z_max = ztest.min(), ztest.max()
        fig, ax = plt.subplots(figsize =(15,10))
        c = ax.pcolormesh(x1test, x2test, ztest, cmap='RdBu', vmin=z_min, vmax=z_max)
        ax.set_title('Test surface')
        ax.axis([x1test.min(), x1test.max(), x2test.min(), x2test.max()])
        fig.colorbar(c, ax=ax)
        plt.show()

        # Transforming generating dimensions to draw noise dimensions
        x1 = x1.reshape(ntrain,)
        x2 = x2.reshape(ntrain,)
        x1test = x1test.reshape(ntest,)
        x2test = x2test.reshape(ntest,)
        X1 = norm.ppf(x1/(1+1e-4)+1e-8/(1+1e-4))
        X2 = norm.ppf(x2/(1+1e-4)+1e-8/(1+1e-4))
        X1test = norm.ppf(x1test/(1+1e-4)+1e-8/(1+1e-4))
        X2test = norm.ppf(x2test/(1+1e-4)+1e-8/(1+1e-4))  
        X = np.column_stack((X1,X2))
        Xtest = np.column_stack((X1test,X2test))
        
        # Drawing noise dimensions using Gaussian conditioning and copulas
        P = np.ones((2,p))*corrxz
        if block_corr:
            P[1,int((p)/2):]=0
            P[0,:int((p)/2)]=0
            length_group1 = len(P[0,:int((p)/2)])
            length_group2 = len(P[0,int((p)/2):])
            Sigma1 = np.diag(np.ones(length_group1))*(1-corrzz)+corrzz
            Sigma2 = np.diag(np.ones(length_group1))*(1-corrzz)+corrzz
            Sigma = np.block([[Sigma1, np.zeros((length_group1, length_group2))],
                             [np.zeros((length_group2, length_group1)), Sigma2]])
        else:
            Sigma = np.diag(np.ones(p))*(1-corrzz)+corrzz
        Covmat = Sigma - P.T @ P
        var_x = np.var(X1)
        var_xtest = np.var(X1test)
        Z = np.random.multivariate_normal(np.zeros(p), var_x*Covmat, ntrain)+X @ P    
        Z = norm.cdf(Z)
        Ztest = np.random.multivariate_normal(np.zeros(p), var_xtest*Covmat, ntest)+Xtest @ P    
        Ztest = norm.cdf(Ztest)
        
        # Joining and returning dataset
        X = np.row_stack((np.column_stack((x1,x2,Z)),np.column_stack((x1test,x2test,Ztest))))
        select = (np.linspace(0,p+1,p+2)<2)
        return Y,F,X,e,sigma,select,ntrain,ntest
    
    
class train:
    
    def get_NN(y,X,xtest,l,s,kern,NN,fraction):
        
        n = len(X)
        
        if fraction <= (NN+1)/n:
             nn = np.random.choice(n,NN,False)
    
        if fraction<1:
            shuffled_index = np.random.choice(n,int(n*fraction),False)
            Xsubset = X[shuffled_index]
            ysubset = y[shuffled_index]
            dists = ((xtest-Xsubset)**2*l**2).sum(1)
            nn = shuffled_index[np.argsort(dists)[:NN]]
            
        else:
            dists = ((xtest-X)**2*l**2).sum(1)
            nn = np.argsort(dists)[:NN]
        dists = sorted(dists)[:NN]   

        return dists, nn
    
    def get_minibatch(y,X,l,s,sigma,reg,minibatch,kern,sampling_strat,nn_fraction):
        
        n,p= np.shape(X)
        select = l!=0

        if sampling_strat !="unif":
            
            # Getting NN set
            i = int(np.random.choice(n,1,False))
            dist, samples = train.get_NN(y,X[:,select],X[i,select], l[select], s, kern, minibatch, nn_fraction)
        else:    
            # Getting minibatch of data
            samples=np.random.choice(n,minibatch,False)
        y_sample=y[samples]
        X_sample=X[samples]

        # Computing kernel matrices
        K=kernel_funcs.ARD(l,s,X_sample,kern)
        R=np.eye(minibatch)*(reg+sigma**2)      
        Ktild=K+R

        return y_sample, X_sample, K, Ktild
    
    def get_gradients_gp_ss(y_sample,X_sample,K,Ktild,l,s,sigma,v,c,lmbda,minibatch,grad_kern,n):
            
            p = len(l)
            
            # Getting common term in Marginal log likelihood gradient (Kinv_y)(Kinv_y)^T-K_inv
            A=model_funcs.grad_logL_A(y_sample,Ktild)
    
            # Gradient wrt l
            grad_logL_l=np.zeros(p)
            for i in range(p):
                Xsample_i=X_sample[:,i][:,None]
                g=grad_kern(K=K,X=Xsample_i,l=l[i],s=s)
                grad_logL_l[i]=0.5*np.sum(A*g)-v*l[i]*(c*lmbda[i]+(1-lmbda[i]))*minibatch/n
                    
            # Gradient wrt s
            g=kernel_funcs.gradK_s(K,s)  
            grad_logL_s=0.5*np.sum(A*g)
                    
            # Gradient wrt sigma
            g=np.eye(minibatch)*sigma*2
            grad_logL_sig=0.5*np.sum(A*g)
            
            return grad_logL_l , grad_logL_s, grad_logL_sig
            
        
    def get_step_size(new_gradient, sum_sq_grads, sum_grads, beta, beta2, eps, learn_rate, optimisation, minibatch, t, n):
            
        if optimisation == "adam":
            sum_sq_grads = (1-beta2)*new_gradient**2+beta2*sum_sq_grads
            sum_grads = (1-beta)*new_gradient+beta*sum_grads
            step_size = learn_rate*(sum_grads/(1-beta**t))/((sum_sq_grads/(1-beta2**t))**0.5+eps)
       
        elif minibatch<n:
            step_size = learn_rate/t*new_gradient
        
        else:
            step_size = learn_rate*new_gradient
            
        return step_size, sum_sq_grads, sum_grads


    def kernel_param_optimise(y,X,l0=0.1,s0=1,sig0=1,v=1e+4,c=1e-8,lmbda=1,reg=1e-3,kern=kernel_funcs.gaussian,grad_kern=kernel_funcs.grad_gaussian,minibatch=256,sampling_strat="nn",nn_fraction=1,
            optimisation="adam",learn_rate=0.01,beta=0.9,beta2=0.999,eps=1e-8,optimisation_sums=[],maxiter=200,print_=False,store_ls=False, L=[], iters=0,store_elbo=False):
        """
        
        Function to optimise q(\theta) and (s,sigma) in a-CAVI
        
        Parameters
        ----------
        y : output vector (n x 1)
        X : input matrix (n x p)
        l0 : initial inverse-lengthscale (post mean - ZT)
        s0 : initial scale
        sig0 : initial noise
        c : spike and slab divergence parameter
        lmbda : < binary inclusion variables > 
        reg : nugget regularisation
        kern :  Kernel function
        grad_kern : Kernel funtion gradient wrt. i^th inverse lengthscale
        minibatch : #samples to draw in SGD (without replacement)
        sampling_strat : "unif" for uniform, "nn" (or any word) for NN
        nn_fraction : Float \in (0,1], fraction of data for stochastic NN search 
        optimisation : "adam" for ADAM, else use SGD with RM sequence if m<n, else use fixed learn rate
        learn_rate : base learning rate
        beta : ADAM retention factor (first moments)
        beta2 : ADAM retention factor (second moments)
        eps : ADAM epsilon imprecision
        optimisation_sums : initial sum of gradients and sum of squared gradients
        maxiter : # gradient steps
        print_ : True = print log likelihood lower bound
        store_ls : True = store inverse lengthscales at each step
        L : previously stored inverse lengthscales
        
        Returns
        -------
        post. mean inverse lengthscales: l 
        scale param: s)
        noise param: sigma 
        Elbo component: logl 
        ADAM exponential MAs: optimisation_sums
        """
        
        plt.rcParams.update({'text.color' : "black",
                      'xtick.color' : "black",
                      'ytick.color' : "black",
                     'axes.labelcolor' : "black"})
        
        n,p = np.shape(X)
    
        # Initialising parameters
        l=np.ones(p)*l0
        lmbda=np.ones(p)*lmbda
        s=s0
        sigma=sig0
        if not np.any(L):
            L = np.zeros(p)[None,:]
            
        # Getting optimisation sums
        if not optimisation_sums:
            sum_sq_grads_l, sum_grads_l,sum_sq_grads_s, sum_grads_s, sum_sq_grads_sig, sum_grads_sig = np.zeros(p),np.zeros(p),0,0,0,0
        else:
            sum_sq_grads_l, sum_grads_l,sum_sq_grads_s, sum_grads_s, sum_sq_grads_sig, sum_grads_sig = optimisation_sums

        # Getting initial minibatch
        minibatch = int(minibatch)
        y_sample, X_sample, K, Ktild = train.get_minibatch(y,X,l,s,sigma,reg, minibatch,kern, sampling_strat, nn_fraction)
              
        # Commencing gradient steps
        t=0
        logLs=np.zeros(maxiter)
        while t<maxiter:
            
            t+=1
            
            # Getting gradients
            grad_logL_l,grad_logL_s,grad_logL_sig = train.get_gradients_gp_ss(y_sample,X_sample, K, Ktild, l, s, sigma, v, c, lmbda, minibatch,grad_kern=grad_kern, n=n)
            
            # Getting step sizes
            step_size_l, sum_sq_grads_l, sum_grads_l = train.get_step_size(grad_logL_l, sum_sq_grads_l, sum_grads_l, beta, beta2, eps, learn_rate, optimisation, minibatch, t+maxiter*iters, n)
            step_size_s, sum_sq_grads_s, sum_grads_s = train.get_step_size(grad_logL_s, sum_sq_grads_s, sum_grads_s, beta, beta2, eps, 0.01, optimisation, minibatch, t+maxiter*iters, n)
            step_size_sig, sum_sq_grads_sig, sum_grads_sig = train.get_step_size(grad_logL_sig, sum_sq_grads_sig, sum_grads_sig, beta, beta2, eps, 0.01, optimisation, minibatch, t+maxiter*iters, n)
            
            # Optional Elbo update
            if store_elbo:
                logLs[t-1]=model_funcs.logL(y_sample,Ktild)*n/minibatch-(0.5*np.sum(v*(c*lmbda+(1-lmbda))*l**2-lmbda*np.log(c))+p/2*np.log(v))*(v!=c)

            # Taking SGD step
            l+=step_size_l
            if kern!="lin":
                s+=step_size_s
            sigma+=step_size_sig
            
            # Updating and printing logL if required
            if print_ and not t % 10:
                if store_ls:
                    L = np.append(L,l.reshape(1,p), axis=0)
                    clear_output(wait=True)        
                    fig,axs = plt.subplots(figsize=(10,10))
                    for i in np.arange(p)[::-1]:
                        lines0 = axs.plot(np.abs(L[:,i]), color = "orange", linewidth = 1)
                    plt.show()
                else:
                    clear_output(wait=True)
                    fig,axs = plt.subplots(figsize=(10,10))
                    axs.plot(logLs[:t], color = "green", linewidth = 1)
                    plt.show()
            
            # Subsampling for next step
            y_sample, X_sample, K, Ktild = train.get_minibatch(y,X,l,s,sigma,reg, minibatch, kern, sampling_strat, nn_fraction)
        
        # returning final params and logL 
        return l,s,sigma,logLs,L,[sum_sq_grads_l, sum_grads_l,sum_sq_grads_s, sum_grads_s, sum_sq_grads_sig, sum_grads_sig]
    
    # CAVI update for q(gamma)
    def get_lmbda(l,v,c,logpi,log1_pi):
        return 1/(1+(1/c)**0.5*np.exp(-0.5*l**2*v*(1-c)+log1_pi-logpi))
    
    # CAVI update for q(pi)
    def get_pi(lmbda,alpha,beta):
        a=(np.sum(lmbda)+alpha)
        b=(len(lmbda)-np.sum(lmbda)+beta)
        elogpi=sp.digamma(a)-sp.digamma(a+b)
        elog1_pi=sp.digamma(b)-sp.digamma(a+b)
        return elogpi, elog1_pi, a, b
    
    # a-CAVI algorithm
    def aCAVI(y,X,l0=0.1,s0=1,sig0=1,lmbda0=1,logpi0=0,log1_pi0=0,v=1e+4,c=1e-8,a=1e-3,b=1e-3,reg=1e-2,aCAVI_iter=5,
                            init_grad_step=200,grad_step=100,minibatch=256,sampling_strat = "nn", nn_fraction = 1,
                            optimisation="adam",learn_rate=0.01,Beta=0.9,Beta2=0.999,eps=1e-8,print_gradsteps=False,print_aCAVI=False,
                            timer=False,store_elbo=False,kern=kernel_funcs.gaussian,grad_kern = kernel_funcs.grad_gaussian, store_ls = False, 
                             seed = [],prune = True, final_prune = False, prune_PIP=0.5, learn_scale = False, optimisation_sums = []):
        """
        a-CAVI training algorithm
        
        Parameters
        ----------
        y : output vector (n x 1)
        X : input matrix (n x p)
        l0 : initial value of inverse length-scales
        s0 : initial value of global scale
        sig0 : initial noise variance
        lmbda0 : initial binary inclusion probabilities   
        logpi0 : initial < log pi >
        log1_pi0 : initial < log 1-pi >
        v : spike and slab scale parameter 
        c : spike and slab divergence parameter
        a : prior hyperparameter on p(pi)
        b : prior hyperparameter on p(pi)
        reg : nugget regularisation
        aCAVI_iter : # VBEM iterations
        init_grad_step : # gradient steps for first a-CAVI iteration
        grad_step : # gradient steps for second + a-CAVI iterations
        minibatch : # SGD samples to draw
        sampling_strat : "unif" for uniform, "nn" (or any word) for NN
        nn_fraction : Float \in (0,1], fraction of data for stochastic NN search 
        ELBO_sample : # samples to use in ELBO update
        optimisation : "adam", "amsgrad", or "sgd"
        learn_rate : SGD learning rate
        Beta : adam/amdgrad E(g) retention factor
        Beta2 : adam/amsgrad E(g2) retention factor
        eps : adam/amsgrad epsilon imprecision
        print_gradsteps : True = print relevant ELBO component per each gradient step
        print_aCAVI : True = print ELBO at each a-CAVI iteration
        timer : True = Time
        kern : Kernel function
        grad_kern : Kernel function gradient
        store_ls : True = Store and plot inverse-lengthscales at each iterations
        seed : random seed for minibatch draws
        prune : Boolean, True = iteratively remove variables with low PIP after every aCAVI iteration
        final_prune : Boolean, True = PIP pruning threshold after final aCAVI iteration
        prune_PIP : pruning threshold
        learn_scale: Boolean, True = learn scale parameter v (DO NOT USE)
        
        Returns
        -------
        post. mean inverse lengthscales : l 
        scale param : s 
        noise param : sig 
        PIPs : lmbda 
        spike and slab scale : v 
        q(pi) stats : [logpi,log1_pi,ahat,bhat]
        ELBOs : Elbos
        """
        
        # Setting dimensions and timer (and computing distance matrix if required)
        if timer:
            t=time.time()
        n,p = np.shape(X)
                
        # Initialising parameters
        l=l0*np.ones(p)
        L = l.reshape(1,p)
        lmbda=lmbda0*np.ones(p)
        ahat,bhat = 1,1
        logpi=logpi0
        log1_pi=log1_pi0
        s = s0*np.var(y)**0.5
        sig = sig0*np.var(y)**0.5
            
        # Filling in missing parameter settings
        if n<minibatch:
            minibatch = n
        if seed:
            np.random.seed(seed)
        if not optimisation_sums:
            optimisation_ssgs = [np.zeros(p),np.zeros(p),0,0,0,0]
        else:
            optimisation_ssgs = optimisation_sums*1

        # Initialisating Elbos and convergence criteria
        i=0
        Elbos=np.zeros(1)
        
        # Initialising selections
        select = np.repeat(True,p)
        latest_select_out = np.repeat(True,p)
        
        # Running VBEM iterations
        while i<aCAVI_iter:
            
            if i==0:
                max_iter = init_grad_step
            if i==1:
                max_iter = grad_step
            
            if max_iter>0:
                # optimising psi, phi
                l[select],s,sig,logLs,L,optimisation_ssgs=train.kernel_param_optimise(
                            y,X[:,select],l0=l[select],s0=s,sig0=sig,reg=reg,minibatch=minibatch,sampling_strat = sampling_strat, nn_fraction = nn_fraction,
                            optimisation=optimisation,learn_rate=learn_rate,beta=Beta,beta2=Beta2,eps=eps,optimisation_sums = optimisation_ssgs,
                            v=v,c=c,lmbda=lmbda[select],maxiter=max_iter,print_=print_gradsteps,kern=kern,grad_kern=grad_kern, store_ls=store_ls, L=L, iters=i,store_elbo=store_elbo)
            else:
                logLs=np.zeros(1)
                
            # Getting ELbo retrospectively
            if store_elbo:
                reg_lmbda=lmbda*1
                reg_lmbda[reg_lmbda>1-1e-10]=1-1e-10
                reg_lmbda[reg_lmbda<1e-10]=1e-10
                neg_kl_gamma = np.sum(reg_lmbda*(logpi-np.log(reg_lmbda))+(1-reg_lmbda)*(log1_pi-np.log(1-reg_lmbda)))
                neg_kl_pi = (a-ahat)*logpi + (b-bhat)*log1_pi+sp.gammaln(ahat)+sp.gammaln(bhat)-sp.gammaln(ahat+bhat)
                neg_kl_theta = 0.5*np.sum(lmbda[select==False])*np.log(c)+0.5*np.sum(select==False)*np.log(v)                                                                                                
                Elbos = np.append(Elbos,logLs+neg_kl_gamma+neg_kl_pi+neg_kl_theta)
            
            # Optimising q(gamma),q(pi)
            lmbda[select+latest_select_out]=train.get_lmbda(l[select+latest_select_out],v,c,logpi,log1_pi)
            logpi,log1_pi,ahat,bhat=train.get_pi(lmbda,alpha=a,beta=b)
            
            # Optional scale parameter learning
            if learn_scale:
                v = p/np.sum(l**2*(1-lmbda+lmbda*c))
            
            # Dropout pruning
            if prune:
                selectnew = lmbda>=prune_PIP
                selectnew_subset = lmbda[select]>=prune_PIP
                L = L[:,selectnew_subset]
                l[selectnew==False]=0
                latest_select_out = select!=selectnew
                select = selectnew
                optimisation_ssgs[0],optimisation_ssgs[1] = optimisation_ssgs[0][selectnew_subset],optimisation_ssgs[1][selectnew_subset]
              
            # Printing updates
            i += 1
            if print_aCAVI:
                print("iteration {0}, elbo = {1}".format(i,Elbo))
         
        # Pruning at last stage if final_prune
        if final_prune:
            l[lmbda<prune_PIP]=0
        
        if timer:
            print("run time is :", time.time()-t)

        return l,s,sig,lmbda,v,[logpi,log1_pi,ahat,bhat], Elbos
       
    def model(y, X, hyper_arg=["v"], hyper_vals = [10**4 * 2**np.linspace(np.log2(1000), -np.log2(1000),11)],training_args=[], training_arg_vals=[]):
        """
        
        Runs a-CAVI for varying v
        
        Parameters
        ----------
        y : output vector (n x 1)
        X : input matrix (n x p)
        hyper_arg : hyperparameter argument name in training algorithm
        hyper_vals : list of hyperparameter values to be iterated over
        training algorithm : algorithm for training,
        training_args : list of arguments for training algorithm to be changed from defaults
        training_arg_vals : list of argument values for training algorithm to be changed from defaults
        
        Returns
        -------
        a-CAVI output list: Results
        # selected vars : Selections
        
        Make sure to feed in v in descending order.
        """

        n,p = np.shape(X)
        y = y.reshape(n,1)

        # Setting up storage object
        Loss = np.zeros(len(hyper_vals[0]))

        # Getting master arguments of algorithms, and updating based on non-default values provided
        training_master_args = inspect.getfullargspec(train.aCAVI)[0]
        training_master_arg_defaults = inspect.getfullargspec(train.aCAVI)[3]
        
        # Updating arguments of training algorithm with non-defaults specified
        function_args = list(training_master_arg_defaults)
        if training_args:
            for j in range(len(training_args)):
                index = np.where(training_args[j]==np.array(training_master_args))[0][0]-2
                function_args[index] = training_arg_vals[j]

        Results = []
        Selections = np.zeros(len(hyper_vals[0]))
        selections = 1
                
        # Looping over spike values
        for i in range(len(hyper_vals[0])):
            
            if selections>0:

                # Updating training args based on hyperopt value
                current_training_args = [y,X]+function_args
                for j in range(len(hyper_arg)):
                    hyper_index =  np.where(hyper_arg[j]==np.array(training_master_args))[0][0]
                    current_training_args[hyper_index] = hyper_vals[j][i]
                current_training_args = tuple(current_training_args)

                # Running training algorithm
                Results.append(train.aCAVI(*current_training_args))

                # Determine active selections
                selections = np.sum(Results[i][0]!=0)
                Selections[i] = selections

                # Compute predictive distribution or MSE
                Loss[i] = -Results[i][len(Results[i])-1]

            else:
                Loss[i] = np.max(Loss)
                Results.append(Results[i-1])
                Selections[i] = Selections[i-1]

        return Results,Selections
                
class evaluation:
                
    def Burkner_LOOLPD(y,X,results,kern,reg,regvar,post_var):

        n,p = np.shape(X)
        y = y.reshape(n,1)
        
        select = results[0]!=0
        l = results[0][select]
        Xselect = X[:,select]
        Kxx = kernel_funcs.ARD(l,results[1],Xselect,kern)+(results[2]**2+reg+regvar)*np.eye(n)
        c = np.linalg.inv(np.linalg.cholesky(Kxx))
        A = np.dot(c.T,c)
        a = A @ y
        LOOLPD = np.ones(n)*(-1/2*np.log(2*np.pi))
        Ymean = np.zeros(n)
        Yvar = np.ones(n)
        for i in range(n):
            g = a[i]
            c = A[i,i]
            Ymean[i] = y[i] - g/c
            if post_var:
                Yvar[i] = 1/c
            LOOLPD[i] = -0.5*np.log(Yvar[i])-0.5*(y[i]-Ymean[i])**2/Yvar[i]
            
            loolpd = LOOLPD.sum()
            
            #updated_loolpd = loolpd - 0.5*np.sqrt(((LOOLPD-loolpd/n)**2).sum())
            
        #print(loolpd,updated_loolpd)
            
        return loolpd,Ymean,Yvar    
    
    def model(y,X,Results,reg=1e-2,kern=kernel_funcs.gaussian,NN=[],fraction=1,post_var=True, print_=False,use_tree=False,leaf_size=100,seed=0, perturb_sd = 0,reg_var=1):
        
        """
        
        Function to get LOO-LPD for models
        
        Parameters
        ----------
        y : output vector (n x 1)
        X : input matrix (n x p)
        Results : list of outputs from aCAVI()
        reg : nugget regularisation
        kern : kernel function
        NN : nearest neighbour truncation
        fraction : % of data to search over for NNs
        post_var : True = get variances (else set variances to 1)
        print_ : print LOO-LPD per model
        use_tree : True = use kd or ball tree (kd used if dim(active X's) < 100)
        leaf_size : size of tree leafs
        seed : random seed set
        perturb_sd : peturbation sd for inverse lengthscales for robustness to poor local optima
        reg_var : regularisation for posterior variances for increased robustness to outliers
        
        Returns
        -------
        LOO-LPDs : logpredictives) 
        PIPs : PIP
        Marginal post.mean inverse ls : Mu
        BMA posterior model probs: weights
        LOO-LPD means (n x 1): Ymean
        LOO-LPD vars (n x 1): Yvar
        
        """ 
        
        # Set up
        n,p = np.shape(X) 
        y = y.reshape(n,1)
        logpredictives = np.zeros(len(Results))
        Ymean = np.zeros((n,len(Results)))
        Yvar = np.zeros((n,len(Results)))
        Kalman = np.zeros((n,len(Results)))
        if not NN:
            NN = n-1
        fraction = np.min((np.max((fraction, NN/n)),1))
        
        # Getting log predictives
        for j in range(len(Results)):
            
            results = Results[j]
            l=np.abs(results[0])
            select = l!=0
            q = np.sum(select)
            
            
            if q>0:
                
                # setting seed
                np.random.seed(seed)

                # Transforming inputs
                indexes = select
                p = np.sum(indexes)
                Xsearch = X[:,indexes]*l[indexes]
                Xselect = X[:,select]
                lselect = l[select]
                s = results[1]
                sig = results[2]             
                lpred = lselect*1

                # NN or no NN
                if NN<n:
                    if use_tree:
                        if p<100:
                            tree = neighbors.KDTree(Xsearch,leaf_size)
                        else:
                            tree = neighbors.BallTree(Xsearch,leaf_size)

                    # Looping through datapoints
                    ypostmean,ypostvar = np.zeros(n),np.ones(n)*reg_var
                    for i in range(n):

                        if perturb_sd>0:
                            lpred = lselect+np.random.normal(0,perturb_sd,q)

                        Xsearchi = Xsearch[i]
                        Xi = Xselect[i][None,:]
                        yi = y[i]

                        # Getting NN set using selected dimensions
                    
                        if use_tree:
                            nn = tree.query(Xsearchi[None,:], k=NN+1)[1][0][1:]
                        else:
                            nn = train.get_NN(y,Xsearch,Xsearchi, 1, s, kern, NN+1, fraction=fraction)[1][1:]
                        
                        # Making predictions 
                        K=kernel_funcs.ARD(l=lpred,s=s,X=Xselect[nn],kern=kern)
                        Ktest=kernel_funcs.ARDtest(l=lpred,s=s,X=Xselect[nn],Xtest=Xi,kern=kern)
                        Ktild=K+np.eye(len(K))*(reg+reg_var)#+sig**2) 
                        KtestKtild = np.linalg.solve(Ktild, Ktest.T).T
                        ypostmean[i] += KtestKtild @ y[nn]
                        if post_var:
                            Ktesttest=kernel_funcs.ARD(l=lpred,s=s,X=Xi,kern=kern)
                            ypostvar[i] += Ktesttest -  KtestKtild @ Ktest.T

                        logpredictives[j] +=  -0.5*np.log(ypostvar[i])-0.5*(ypostmean[i]-yi)**2/ypostvar[i]
                    
                    logpredictives[j] += -n/2*np.log(2*np.pi)
                    
                else:
                    logpred,ypostmean,ypostvar = evaluation.Burkner_LOOLPD(y,X,results,kernel_funcs.gaussian,reg,reg_var,post_var)
                    logpredictives[j] += logpred
            else:
                logpredictives[j] = np.min(logpredictives)-1000
                
            #if print_:
            print("LOO-LPD {0} is :".format(j+1), logpredictives[j])
            if q!=0:
                Ymean[:,j]=ypostmean
                Yvar[:,j]=ypostvar
        
        # Getting marginal PIPs et al
        weights = np.exp(logpredictives - np.max(logpredictives))
        weights *= 1/np.sum(weights)
        PIP,Mu = 0,0
        for j in range(len(Results)):
            PIP += Results[j][3]*weights[j]
            Mu += np.abs(Results[j][0])*weights[j]

        return logpredictives,PIP,Mu,weights,Ymean,Yvar
                
class test:
        
    def posterior_predictive(y,X,Xtest,l,s,sig,reg=0.01,kern=kernel_funcs.gaussian, post_var=False, latents = False):
                
        fpost_mean, fpost_var, ypost_mean, ypost_var = 0,0,0,0
        K=kernel_funcs.ARD(l=l,s=s,X=X,kern=kern)
        Ktest=kernel_funcs.ARDtest(l=l,s=s,X=X,Xtest=Xtest,kern=kern)
        if post_var:
            Ktesttest=kernel_funcs.ARD(l=l,s=s,X=Xtest,kern=kern)
        Ktild=K+np.diag(np.ones(len(K)))*(reg+sig**2) 
        ypost_mean=Ktest @ np.linalg.solve(Ktild,y)
        if latents:
            fpost_mean=Ktest @ np.linalg.solve(K+np.diag(np.ones(len(K)))*reg,y)
        if post_var:
            ypost_var=Ktesttest-Ktest @ np.linalg.solve(Ktild,Ktest.T)+np.diag(np.ones(len(Ktesttest)))*(reg+sig**2)
            if latents:
                fpost_var=Ktesttest-Ktest @ np.linalg.solve(K+np.diag(np.ones(len(K)))*reg,Ktest.T)+np.diag(np.ones(len(Ktesttest)))*reg
        
        return fpost_mean, fpost_var, ypost_mean, ypost_var
    
    def posterior_predictive_nn(y,X,Xtest,l,s,sig,reg=0.01,kern=kernel_funcs.gaussian,NN=256,fraction=1,post_var=False, latents = False, print_=True, use_tree=True,leaf_size=100):
    
        # Setting up storage objects
        m,p = np.shape(Xtest)
                    
        fpost_mean=np.zeros((m,1))
        ypost_mean=np.zeros((m,1))
        fpost_var=np.zeros((m,m))
        ypost_var=np.zeros((m,m))
        
        Xsearch=X*l
        Xtestsearch = Xtest*l
        
        # Building tree
        if use_tree:
            if p<100:
                tree = neighbors.KDTree(Xsearch,leaf_size)
            else:
                tree = neighbors.BallTree(Xsearch,leaf_size)
            
        # Getting predictions           
        for i in range(m):

            # Getting NN set
            if use_tree:
                nn = tree.query(Xtestsearch[i][None,:], k=NN)[1][0]
            else:
                nn = train.get_NN(y,Xsearch,Xtestsearch[i], 1, s, kern, NN, fraction=fraction)[1]

            # Updating lengthscales and data to predict with if no gradsteps taken
            Xi = X[nn,:]
            Xtesti = Xtest[i]
            yi=y[nn]

            # Making predictions 
            K=kernel_funcs.ARD(l=l,s=s,X=Xi,kern=kern)
            Ktest=kernel_funcs.ARDtest(l=l,s=s,X=Xi,Xtest=Xtesti[None,:],kern=kern)
            if post_var:
                Ktesttest=kernel_funcs.ARD(l=l,s=s,X=Xtesti[None,:],kern=kern)
            Ktild=K+np.diag(np.ones(len(K)))*(reg+sig**2) 

            KtestK = np.linalg.solve(K+np.diag(np.ones(len(K)))*reg, Ktest.T).T
            KtestKtild = np.linalg.solve(Ktild, Ktest.T).T
            if latents:
                fpost_mean[i] = KtestK @ yi
            ypost_mean[i] = KtestKtild @ yi
            if post_var:
                if latents:
                    fpost_var[i,i] = Ktesttest-KtestK @ Ktest.T
                ypost_var[i,i] = Ktesttest- KtestKtild @ Ktest.T
            if print_:
                if not round(i/m*100,2) % 10:
                    print(i/m*100, "% complete")


        return fpost_mean, fpost_var, ypost_mean, ypost_var
    
    def model(y,X,Xtest,testing_algorithm, Results,weights, MC_samples=100):
        
        n,m = len(X),len(Xtest)
        y = y.reshape(n,1)
        
        # Getting MC_samples to discretise weights
        sampled_weights = np.random.multinomial(MC_samples,weights, 1)[0]/MC_samples
        
        # Getting preditions per model
        fmean,ymean = 0,0
        
        for i in range(len(weights)):
            
            if sampled_weights[i]>0:
            
                l = Results[i][0]
                s = Results[i][1]
                sig = Results[i][2]
                select = l!=0
                q = np.sum(select)

                fm, fv, ym, yv = testing_algorithm(y=y, X=X[:,select].reshape(n,q), Xtest = Xtest[:,select].reshape(m,q), l=l[select], s=s,sig=sig, post_var=False)

                fmean += sampled_weights[i]*fm
                ymean += sampled_weights[i]*ym
        
        return fmean, ymean

class diagnostics:
    
    # normalised MSE
    def MSE_pc(x,y):
        assert np.all(np.shape(x) == np.shape(y))
        return np.sum((x-y)**2)/np.sum((y-y.mean())**2)