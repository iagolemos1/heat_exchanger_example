# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 10:24:10 2020

@author: iago
"""
import warnings
import math as mt
import numpy as np
import threshmodels2 as tm2

import scipy.special as sm
from scipy.stats import norm
from scipy.stats import chi2

import numdifftools as nd
import matplotlib.pyplot as plt

from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
POTr = importr('POT')

warnings.filterwarnings('ignore')

class build_model(object):
    """
    Builds a GPD model by fitting the GPD to a given dataset.
    
    Parameters
    ----------
    data: Numpy array with containing the sample to fit;
    
    threshold: Float with the threshold value;
    
    method: Fit method;
    
    alpha: Significance level for computing estimate standard errors. Default is 0.05;
      
    kwargs: Other arguments for the other functions;    
    """
    
    def __init__(self, data, threshold, method = 'mle', alpha = 0.05, **kwargs):
        self.sample = data
        self.method = method
        self.u = threshold
        self.n = len(data)
        
        rdata = FloatVector(data)
        y = []
        for x in data:
            if x > self.u:
                y.append(x - self.u)
        
        y = np.array(y).reshape((1, -1))
        k = y.shape[1]
        
        if method == 'mle' or method == 'mple':
            
            self.res = POTr.fitgpd(rdata, threshold, est = method)
            
            if self.res[8][0] == 'successful':
            
                self.f_eval = self.res[9][0]
                self.g_eval = self.res[9][1]
                self.f_val = self.res[19][0]
                self.scale = self.res[0][0]
                self.shape = self.res[0][1]
                self.m = self.res[12][0]
                self.proportion = self.res[13][0]
                
                def f(y,x):
                    return((1/x[0])*((1 + y*x[1]/x[0])**-(1/x[1] + 1)))
                               
                def log(x):
                    return(np.sum(np.log(f(y,x))))
                
                self.var_cov = -1*np.linalg.inv(nd.Hessian(log)([self.scale, self.shape]))
                
                z = norm.ppf(1 - alpha/2)
                
                self.std_errors = np.array([self.var_cov[0,0]**0.5, self.var_cov[1,1]**0.5])
                
                self.scale_variation = [self.scale - z * self.std_errors[0], self.scale + z * self.std_errors[0]] 
                self.shape_variation = [self.shape - z * self.std_errors[1], self.shape + z * self.std_errors[1]] 
                self.Deviance = -2*log([self.scale, self.shape])
                self.AIC = self.Deviance + 4
                self.BIC = -2*log([self.scale, self.shape]) + 4*mt.log(k)
            
            else: 
                print('Convergence not reached.')
            
        elif method == 'lme':
            self.res = POTr.fitgpd(rdata, threshold, est = method)
            self.f_eval = self.res[10][0]
            self.g_eval = self.res[10][1]
            self.scale = self.res[0][0]
            self.shape = self.res[0][1]
            self.m = self.res[11][0]
            self.proportion = self.res[12][0]
            self.std_errors = np.array([self.res[1][0], self.res[1][1]])
            z = norm.ppf(1 - alpha/2)
            self.scale_variation = [self.scale - z * self.std_errors[0], self.scale + z * self.std_errors[0]] 
            self.shape_variation = [self.shape - z * self.std_errors[1], self.shape + z * self.std_errors[1]] 
            self.corr = np.array([[self.res[8][0],self.res[8][1]],[self.res[8][2],self.res[8][3]]])

        elif method == 'pwmb':
            if 'a' in kwargs:
                a = kwargs['a']
            else:
                a = 0.35
            
            if 'b' in kwargs:
                b = kwargs['b']
            else:
                b = 0
            
            if 'hybrid' in kwargs:
                h = kwargs['hybrid']
            else:
                h = False

            self.res = POTr.fitgpd(rdata, threshold, est = method, a = float(a), b = float(b), hybrid = h)
            self.scale = self.res[0][0]
            self.shape = self.res[0][1]     
            self.std_errors = np.array([self.res[1][0], self.res[1][1]])
            self.m = self.res[9][0]
            self.proportion = self.res[10][0]
            z = norm.ppf(1 - alpha/2)
            self.scale_variation = [self.scale - z * self.std_errors[0], self.scale + z * self.std_errors[0]] 
            self.shape_variation = [self.shape - z * self.std_errors[1], self.shape + z * self.std_errors[1]]
            self.f_eval = self.res[7][0]
            self.g_eval = self.res[7][0]
            self.corr =  np.array([[self.res[6][0],self.res[6][1]],[self.res[6][2],self.res[6][3]]])
        
        elif method == 'pwmu':
            if 'hybrid' in kwargs:
                h = kwargs['hybrid']
            else:
                h = False
                
            self.res = POTr.fitgpd(rdata, threshold, est = method, hybrid = h)
            self.scale = self.res[0][0]
            self.shape = self.res[0][1]     
            self.std_errors = np.array([self.res[1][0], self.res[1][1]])
            self.m = self.res[9][0]
            self.proportion = self.res[10][0]
            z = norm.ppf(1 - alpha/2)
            self.scale_variation = [self.scale - z * self.std_errors[0], self.scale + z * self.std_errors[0]] 
            self.shape_variation = [self.shape - z * self.std_errors[1], self.shape + z * self.std_errors[1]]
            self.f_eval = self.res[7][0]
            self.g_eval = self.res[7][0]
            self.corr =  np.array([[self.res[6][0],self.res[6][1]],[self.res[6][2],self.res[6][3]]])
        
        elif method == 'moments':
            self.res = POTr.fitgpd(rdata, threshold, est = method)
            self.scale = self.res[0][0]
            self.shape = self.res[0][1]     
            self.std_errors = np.array([self.res[1][0], self.res[1][1]])
            self.m = self.res[6][0]
            self.proportion = self.res[7][0]
            z = norm.ppf(1 - alpha/2)
            self.scale_variation = [self.scale - z * self.std_errors[0], self.scale + z * self.std_errors[0]] 
            self.shape_variation = [self.shape - z * self.std_errors[1], self.shape + z * self.std_errors[1]]
            self.f_eval = self.res[10][0]
            self.g_eval = self.res[10][0]
            self.corr =  np.array([[self.res[9][0],self.res[9][1]],[self.res[9][2],self.res[9][3]]])
        
        elif method == 'pickands':
            self.res = POTr.fitgpd(rdata, threshold, est = method)
            self.scale = self.res[0][0]
            self.shape = self.res[0][1] 
            self.m = self.res[6][0]   
            self.proportion = self.res[7][0]
            
        
        elif method == 'med':
            if 'tol' in kwargs:
                tol = kwargs['tol']
            else:
                tol = 1e-3
            
            if 'maxit' in kwargs:
                maxit = kwargs['maxit']
            else:
                maxit = 500
            
            self.res = POTr.fitgpd(rdata, threshold, est = method, tol = tol, maxit = maxit)   
            self.scale = self.res[0][0]
            self.shape = self.res[0][1] 
            self.m = self.res[12][0]   
            self.proportion = self.res[13][0]
            self.f_eval = int(self.res[9][0])
            self.convergence = self.res[8][0]
        
        elif method == 'mdpd':
            if 'a' in kwargs:
                a = kwargs['a']
            else:
                a = 0.1
                
            self.res = POTr.fitgpd(rdata, threshold, est = method, a = a)
            self.scale = self.res[0][0]
            self.shape = self.res[0][1] 
            self.m = self.res[12][0]   
            self.proportion = self.res[13][0]
            self.f_eval = int(self.res[9][0])
            self.g_eval = int(self.res[9][1])
            self.convergence = self.res[8][0]

        elif method == 'mgf':
            stat = kwargs['stat']
            self.res = POTr.fitgpd(rdata, threshold, est = method, stat = stat)
            self.scale = self.res[0][0]
            self.shape = self.res[0][1] 
            self.m = self.res[11][0]   
            self.proportion = self.res[12][0]
            self.f_eval = int(self.res[8][0])
            self.g_eval = int(self.res[8][1])
            self.convergence = self.res[7][0]
            
            
    def disp(self):
        """
        Prints model estimatives and parameters.
        """
        if self.method == 'mle' or self.method == 'mple':
            print('\nModel Estimatives and Parameters\n')
            print('--------------')
            print('Threshold: {}\nNumber above: {}\nProportion Above: {:.4f}'.format(self.u, self.m, self.proportion))
            print('--------------')
            print('Estimator:',self.method,'\n\nEstimates\nScale parameter: {:.4f} [{:.4f},{:.4f}]\nShape parameter: {:.4f} [{:.4f},{:.4f}]'.format(self.scale, self.scale_variation[0], self.scale_variation[1], self.shape, self.shape_variation[0], self.shape_variation[1]))
            print('\nStandard errors\nScale parameter: {:.4f}\nShape parameter: {:.4f}'.format(self.std_errors[0], self.std_errors[1]))
            print('--------------')
            print('Metrics')
            print('\nDeviance: {:.4}'.format(self.Deviance))
            print('AIC: {:.4}'.format(self.AIC))
            print('BIC: {:.4}'.format(self.BIC))
            print('--------------')
            print('Optimization Information')
            print('\nConvergence: Successful')
            print('Log-likelihood value: {:.4}'.format(self.f_val))
            print('Function Evaluations: {}'.format(self.f_eval))
            print('Gradient Evaluations: {}'.format(self.g_eval))
        
        elif self.method == 'lme' or self.method == 'pwmb' or self.method == 'pwmu' or self.method == 'moments':
            print('\nModel Estimatives and Parameters\n')
            print('--------------')
            print('Threshold: {}\nNumber above: {}\nProportion Above: {:.4f}'.format(self.u, self.m, self.proportion))
            print('--------------')
            print('Estimator:',self.method,'\n\nEstimates\nScale parameter: {:.4f} [{:.4f},{:.4f}]\nShape parameter: {:.4f} [{:.4f},{:.4f}]'.format(self.scale, self.scale_variation[0], self.scale_variation[1], self.shape, self.shape_variation[0], self.shape_variation[1]))
            print('\nStandard errors\nScale parameter: {:.4f}\nShape parameter: {:.4f}'.format(self.std_errors[0], self.std_errors[1]))
            print('--------------')
            print('Optimization Information')
            if self.method == 'lme':
                print('\nConvergence: Successful')
            print('Function Evaluations: {}'.format(self.f_eval))
            print('Gradient Evaluations: {}'.format(self.g_eval))    
            
        
        elif self.method == 'pickands':
            print('\nModel Estimatives and Parameters\n')
            print('--------------')
            print('Threshold: {}\nNumber above: {}\nProportion Above: {:.4f}'.format(self.u, self.m, self.proportion))
            print('--------------')
            print('Estimator:',self.method,'\n\nEstimates\nScale parameter: {:.4f}\nShape parameter: {:.4f}'.format(self.scale, self.shape))
            print('--------------')
            print('Optimization Information')
            print(self.res[4][0])
                
        
        elif self.method == 'med' or self.method == 'mdpd' or self.method == 'mgf':
            print('\nModel Estimatives and Parameters\n')
            print('--------------')
            print('Threshold: {}\nNumber above: {}\nProportion Above: {:.4f}'.format(self.u, self.m, self.proportion))
            print('--------------')
            print('Estimator:',self.method,'\n\nEstimates\nScale parameter: {:.4f}\nShape parameter: {:.4f}'.format(self.scale, self.shape))
            print('--------------')
            print('Optimization Information')
            print('\n',self.convergence)
            print('Function Evaluations: {}'.format(self.f_eval))
            if self.method == 'mdpd' or self.method == 'mgf':
                print('Gradient Evaluations: {}'.format(self.g_eval))
            
    def estimates(self):
        """
        Returns
        -------
        scale: Scale parameter value;
        
        shape: Shape parameter value;
        
        std_errors: Scale and Shape parameter standard errors.
        """
        return(self.scale, self.shape)
    
    def corr(self):
        """
        Returns
        -------
        corr: Correlation matrix of the estimates. Available for lme, pwmu, pwmb and momenths methods.
        """
        return(self.corr)
    
    def std_errors(self):
        """
        Returns
        -------
        std_errors: Scale and Shape parameter standard errors.
        """
        return(self.std_erros)
    
    def var_cov(self):
        """
        Returns
        -------
        var_cov: Variance-Covariance matrix.

        """
        return(self.var_cov)
    
    def metrics(self):
        """
        Just available if the method is mle-based. The metrics are not computed for other fitting methods.

        Returns
        -------
        Deviance: Log-likelihod deviance;
        
        AIC: Akaike information criterion;
            
        BIC: Bayesian information criterion;

        """
        return(self.Deviace, self.AIC, self.BIC)
    
    def data(self):
        """
        Returns
        -------
        sample: Data used for fitting.
        u: threshold value
        """
        return(self.sample, self.u)
    
    def pat(self):
        return(self.proportion)


class gpd_model(object):
    def __init__ (self, model = None, **kwargs):
        if model != None:
            self.scale, self.shape = model.estimates()
            self.data, self.u = model.data()
            
        else:
            args = list(kwargs)
            args_val = list(kwargs.values())
            for i in range(0, len(args)):
                if args[i] == 'scale':
                    self.scale = args_val[i]
                elif args[i] == 'shape':
                    self.shape = args_val[i]
                elif args[i] == 'u' or args[i]=='threshold':
                    self.u = args_val[i]
    
    def cdf(self, x):
        scale = self.scale
        shape = self.shape
        u = self.u  
        nparr = np.array([1])
        
        if type(x) == list or type(x) == type(nparr):
            cdf = []
            for data in x:
                if data >= u:
                    cdf.append(1 - (1 + shape*(data - u)/scale)**(-1/shape))
                else:
                    cdf.append(0)
                    
            return(cdf)
        
        else:
            if x >= u:
                F = 1 - (1 + shape*(x - u)/scale)**(-1/shape)
                return(F)
            else:
                return(0)
        
    def surv(self, x):
        return(1 - gpd_model.cdf(self, x))
    
    def pdf(self, x):
        scale = self.scale
        shape = self.shape
        u = self.u  
        nparr = np.array([1])
        
        if type(x) == list or type(x) == type(nparr):
            pdf = []
            for data in x:
                if data >= u:
                    pdf.append((1/scale)*((1 + (data-u)*shape/scale)**-(1/shape + 1)))
                else:
                    pdf.append(0)
            return(pdf)
        
        else:
            if x >= u:
                f = (1/scale)*((1 + (x-u)*shape/scale)**-(1/shape + 1))
                return(f)
            else:
                return(0)
        
    def icdf(self, F):
        scale = self.scale
        shape = self.shape
        u = self.u        
        
        x = u + (scale/shape)*((1-F)**-shape -1)
        return(x)
       
    def rvs(self, n, low = 0.0, high = 1.0):
        uni = np.random.uniform(low = low, high = high, size = n)
        x = gpd_model.icdf(self, uni)
        return(x)
    
    def mean(self):
        scale = self.scale
        shape = self.shape
        u = self.u
        if shape < 1:
            mean = u + scale/(1 - shape)
            return(mean)
        else: 
            print('Shape parameter value is higher than 1. Mean can note be computed.')
            return(np.NaN)
        
    def var(self):
        scale = self.scale
        shape = self.shape
        
        if shape < 0.5:
            var = (scale**2)/((1-shape)**2 * (1 - 2*shape))
            return(var)
        else:
            print('Shape parameter value is higher than 0.5. Variance can not be computed.')
            return(np.NaN)
        
    def median(self):
        scale = self.scale
        shape = self.shape
        u = self.u

        median = u + scale*(2**shape - 1)/shape
        return(median)
    
    def kurtosis(self):
        shape = self.shape
        
        if shape < (1/4):
            kurtosis = (3*(1 - 2*shape)*(2*shape**2 + shape + 3))/((1 - 3*shape)*(1 - 4*shape)) - 3
            return(kurtosis)
        else:
            print('Shape parameter value is higher than 1/4. Kurtosis can not be computed.')
            return(np.NaN)
        
    def skewness(self):
        shape = self.shape
        
        if shape < (1/3):
            skewness = (2*(1 + shape)*(1 - 2*shape)**0.5)/(1 - 3*shape)
            return(skewness)
        else:
            print('Shape parameter value is higher than 1/3. Kurtosis can not be computed.')
            return(np.NaN)
    
    def entropy(self):
        scale = self.scale
        shape = self.shape
        h = np.log(scale) + shape + 1
        return(h)
        
class model_diag(object):
    
    def __init__ (self, model):
        self.data, self.u = model.data()
        self.model = model
        
    def pdf(self, bins = 'sturges', xlabel = 'Data', ylabel = 'PDF', title = None, label = 'Theoretical PDF'):
        sample = self.data
        u = self.u
        
        sample_excess = []
        for data in sample:
            if data > u:
                sample_excess.append(data)
                
        sample_excess 
        x_points = np.arange(0, max(sample), 0.001)
        pdf = gpd_model(self.model).pdf(x_points)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title != None:
            plt.title(title)

        plt.plot(x_points, pdf, color = 'black', label = label)
        plt.hist(sample_excess, bins = bins, density = True, color = 'dimgray') #draw histograms    
        plt.legend()
        plt.show()
    
    def cdf(self, alpha = 0.05, xlabel = 'Data', ylabel = 'CDF', title = None, theo_cdf_label = 'Theoretical CDF', emp_cdf_label = 'Empirical CDF'):
        sample = self.data
        u = self.u
        sample = sorted(sample)
        sample_excess = []
        for data in sample:
            if data > u:
                sample_excess.append(data)
                
        n = len(sample_excess)
        y = np.arange(1,n+1)/n  
        
        i_initial = 0
        n = len(sample)
        for i in range(0, n):
            if sample[i] > u + 0.0001:
                i_initial = i 
                break
        
        F1 = []
        F2 = []
        for i in range(i_initial,len(sample)):
            e = (((mt.log(2/alpha))/(2*len(sample_excess)))**0.5)  
            F1.append(y[i-i_initial] - e)
            F2.append(y[i-i_initial] + e)   
            
        x_points = np.arange(0, max(sample), 0.001)
        cdf = gpd_model(self.model).cdf(x_points)
        
        plt.figure(7)
        plt.plot(x_points, cdf, color = 'black', label=theo_cdf_label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title != None:
            plt.title(title)
        plt.scatter(sorted(sample_excess), y, label= emp_cdf_label, color = 'black', s = 10)
        plt.plot(sorted(sample_excess), F1, linestyle='--', color='black', alpha = 0.5, lw = 0.9, label = 'DKW Confidence Bands')
        plt.plot(sorted(sample_excess), F2, linestyle='--', color='black', alpha = 0.5, lw = 0.9)
        plt.legend()
        plt.show()
        
    def qqplot(self, alpha = 0.05, xlabel = 'Theoretical Quantiles', ylabel = 'Empirical Quantiles', title = None):
        self.scale, self.shape = self.model.estimates()
        self.data, self.u = self.model.data()
        
        scale = self.scale
        shape = self.shape
        sample = self.data
        threshold = self.u
        
        i_initial = 0
        p = []
        n = len(sample)
        sample = np.sort(sample)
        for i in range(0, n):
            if sample[i] > threshold + 0.0001:
                i_initial = i #get the index of the first observation over the threshold
                k = i - 1
                break
    
        for i in range(i_initial, n):
            p.append((i - 0.35)/(n)) #using the index, compute the empirical probabilities by the Hosking Plotting Poistion Estimator.
    
        p0 = (k - 0.35)/(n)    
    
        quantiles = []
        for pth in p:
           quantiles.append(threshold + ((scale/shape)*(((1-((pth-p0)/(1-p0)))**-shape) - 1))) #getting theorecial quantiles arrays

        sample_excess = []
        for data in sample:
            if data > threshold:
                sample_excess.append(data)
                
        n = len(sample_excess)
        y = np.arange(1,n+1)/n #getting empirical quantiles
    
        #Kolmogorov-Smirnov Test for getting the confidence interval
        K = (-0.5*mt.log(alpha/2))**0.5
        M = (len(p)**2/(2*len(p)))**0.5
        CI_qq_high = []
        CI_qq_low = []
        for prob in y:
            F1 = prob - K/M
            F2 = prob + K/M
            CI_qq_low.append(threshold + ((scale/shape)*(((1-((F1)/(1)))**-shape) - 1)))
            CI_qq_high.append(threshold + ((scale/shape)*(((1-((F2)/(1)))**-shape) - 1)))
    
        coef = np.polyfit(quantiles, sorted(sample_excess), 1)
        poly1d_fn = np.poly1d(coef) 
        
        plt.figure(5)
        plt.scatter(quantiles, sorted(sample_excess), color = 'black', s = 10)
        plt.plot(quantiles, poly1d_fn(quantiles), 'k', label = 'Regression line', lw = 2)
        plt.plot(sorted(sample_excess), CI_qq_low, linestyle='--', color='black', alpha = 0.5, lw = 0.8, label = 'KS Confidence Bands')
        plt.legend()
        plt.plot(sorted(sample_excess), CI_qq_high, linestyle='--', color='black', alpha = 0.5, lw = 0.8)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title != None:
            plt.title(title)

        plt.show()
    
    def ppplot(self, alpha = 0.05, xlabel = 'Empirical Probability', ylabel = 'Theoretical Probability', title = None):
        self.scale, self.shape = self.model.estimates()
        self.data, self.u = self.model.data()
        
        sample = self.data
        threshold = self.u
        
        sample_excess = []
        for data in sample:
            if data > threshold:
                sample_excess.append(data)
                
        n = len(sample_excess)
        y = np.arange(1,n+1)/n 
        
        cdf_pp = sorted(gpd_model(self.model).cdf(sample_excess))
  
        i_initial = 0
        sample = sorted(sample)
        n = len(sample)
        for i in range(0, n):
            if sample[i] > threshold + 0.0001:
                i_initial = i
                break
        F1 = []
        F2 = []
        for i in range(i_initial,len(sample)):
            e = (((mt.log(2/alpha))/(2*len(sample_excess)))**0.5)  
            F1.append(y[i-i_initial] - e)
            F2.append(y[i-i_initial] + e)

        coef = np.polyfit(y, cdf_pp, 1)
        poly1d_fn = np.poly1d(coef) 
        
        #Plotting PP
        plt.figure(6)
        plt.scatter(y, cdf_pp, color = 'black', s = 10)
        plt.plot(y, poly1d_fn(y), 'k', label = 'Regression line', lw = 2)
        plt.plot(y, F1, linestyle='--', color='black', alpha = 0.4, label = 'DKW Confidence Bands')
        plt.plot(y, F2, linestyle='--', color='black', alpha = 0.4)
        plt.legend()
        if title != None:
            plt.title(title)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()        
        
class retlev_tool(object):
    def __init__(self, model, alpha = 0.05):

        self.model = model
        self.alpha = alpha
        
    def retlev(self, return_period):
        alpha = self.alpha
        scale, shape = self.model.estimates()
        sample, threshold = self.model.data()
        
        sample_excess = []
        for data in sample:
            if data > threshold:
                sample_excess.append(data)
                
        m = return_period
        Eu = len(sample_excess)/len(sample)
        x_m = threshold + (scale/shape)*(((m*Eu)**shape) - 1)   
        
        d = Eu*(1-Eu)/len(sample)
        e = self.model.var_cov[0,0]
        f = self.model.var_cov[0,1]
        g = self.model.var_cov[1,0]
        h = self.model.var_cov[1,1]
        a = (scale*(m**shape))*(Eu**(shape-1))
        b = (shape**-1)*(((m*Eu)**shape) - 1)
        c = (-scale*(shape**-2))*((m*Eu)**shape - 1) + (scale*(shape**-1))*((m*Eu)**shape)*mt.log(m*Eu)
        CI = (norm.ppf(1-(alpha/2))*((((a**2)*d) + (b*((c*g) + (e*b))) + (c*((b*f) + (c*h))))**0.5))
        
        return(x_m, CI)
    
    def retlev_plot(self, block_size, return_period, xlabel = 'Return Period', ylabel = 'Return Level', title = None):
        alpha = self.alpha
        scale, shape = self.model.estimates()
        sample, threshold = self.model.data()
        
        sample_excess = []
        for data in sample:
            if data > threshold:
                sample_excess.append(data)
                
        sample_excess = sorted(sample_excess)
        sample = sorted(sample)
        
        Eu = len(sample_excess)/len(sample)        
    
        ny = block_size 
        N_year = return_period/block_size
        
        for i in range(0, len(sample)):
            if sample[i] > threshold + 0.0001:
                i_initial = i 
                break
        
        p = np.arange(i_initial,len(sample))/(len(sample)) 
        N = 1/(ny*(1-p))  
        
        d = Eu*(1-Eu)/len(sample)
        e = self.model.var_cov[0,0]
        f = self.model.var_cov[0,1]
        g = self.model.var_cov[1,0]
        h = self.model.var_cov[1,1]
        
        year_array = np.arange(min(N), N_year+0.1, 0.1)
        z_N = []
        CI_z_N_high_year = []
        CI_z_N_low_year = [] 
        for year in year_array:
            z_N.append(threshold + (scale/shape)*(((year*ny*Eu)**shape) - 1))
            a = (scale*((year*ny)**shape))*(Eu**(shape-1))
            b = (shape**-1)*((((year*ny)*Eu)**shape) - 1)
            c = (-scale*(shape**-2))*(((year*ny)*Eu)**shape - 1) + (scale*(shape**-1))*(((year*ny)*Eu)**shape)*mt.log((year*ny)*Eu)
            CIyear = (norm.ppf(1-(alpha/2))*((((a**2)*d) + (b*((c*g) + (e*b))) + (c*((b*f) + (c*h))))**0.5))
            CI_z_N_high_year.append(threshold + (scale/shape)*(((year*ny*Eu)**shape) - 1) + CIyear)
            CI_z_N_low_year.append(threshold + (scale/shape)*(((year*ny*Eu)**shape) - 1) - CIyear)
        
        #Plotting Return Value
        plt.figure(8)
        plt.plot(year_array, CI_z_N_high_year, linestyle='--', color='black', alpha = 0.5, lw = 0.9, label = 'Confidence Bands')
        plt.plot(year_array, CI_z_N_low_year, linestyle='--', color='black', alpha = 0.5, lw = 0.9)
        plt.plot(year_array, z_N, color = 'black', label = 'Theoretical Return Level')
        plt.scatter(N, sorted(sample_excess), color = 'black', s = 10, label = 'Empirical Return Level')
        plt.xscale('log')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title != None:
            plt.title(title)
        plt.legend()
    
        plt.show()

class POT(object):
    def __init__ (self, data, alpha = 0.05):
        self.data = data
        self.alpha = alpha
        
    def mrl(self):
     
        step = np.quantile(self.data, .9975)/45
        threshold = np.arange(0, np.quantile(self.data, .9975), step=step) 
        z_inverse = norm.ppf(1-(self.alpha/2))
    
        mrl_array = [] 
        CImrl = [] 

        for u in threshold:
            excess = [] 
            for data in self.data:
                if data > u:
                    excess.append(data - u) 
            mrl_array.append(np.mean(excess))
            std_loop = np.std(excess) 
            CImrl.append(z_inverse*std_loop/(len(excess)**0.5)) 
    
        CI_Low = [] 
        CI_High = [] 
    
        for i in range(0, len(mrl_array)):
            CI_Low.append(mrl_array[i] - CImrl[i])
            CI_High.append(mrl_array[i] + CImrl[i])
    
        plt.figure(1)
        plt.plot(threshold, mrl_array, color = 'black')
        plt.plot(threshold, CI_Low, '--', color = 'black', linewidth = 0.8)
        plt.plot(threshold, CI_High, '--', color = 'black', linewidth = 0.8)
        plt.xlabel('u [mm]')
        plt.ylabel('Mean Excesses')
        plt.show()  
        
        return(threshold, mrl_array, CI_Low, CI_High)
    
    def parst(self):
        
        step = np.quantile(self.data, .9975)/45
        threshold = np.arange(0, np.quantile(self.data, .9975), step = step)
        
        stdshape = [] 
        shape = []  
        scale = []  
        mod_scale = [] 
        CI_shape = [] 
        CI_mod_scale = [] 
        z = norm.ppf(1-(self.alpha/2)) 
        
        for u in threshold:
            model = build_model(self.data, u, alpha = self.alpha)
    
            shape.append(model.estimates()[1]) 
            scale.append(model.estimates()[0]) 
            
            stdshape.append(model.std_errors[1]) 
            CI_shape.append(model.std_errors[1]*z) 
            
            mod_scale.append(model.estimates()[0] - model.estimates()[1]*u) #getting the modified scale parameter
    
            var_cov = model.var_cov
            Var_mod_scale = (var_cov[0,0] - (u*var_cov[1,0]) - u*(var_cov[0,1] - (var_cov[1,1]*u)))
            CI_mod_scale.append((Var_mod_scale**0.5)*z) 
          
        plt.figure(2)    
        plt.errorbar(threshold, shape, yerr = CI_shape, fmt = 'o' , color = 'black', markerfacecolor='none')
        plt.xlabel('u [mm]')
        plt.ylabel('Shape Parameter')
        

        plt.figure(3)
        plt.errorbar(threshold, mod_scale, yerr = CI_mod_scale, fmt = 'o', color = 'black', markerfacecolor='none')
        plt.xlabel('u [mm]')
        plt.ylabel('Modified Scale Parameter')
        
        
        plt.show()
        
    def diplot(self, n, xlabel = 'u', ylabel = 'Dispersion Index', title = None):
        
        step = np.quantile(self.data, .9975)/60
        threshold = np.arange(0, np.quantile(self.data, .9975), step=step)
        data = self.data.reshape((-1, n))
        n_rows = data.shape[0]
        di = []
        for u in threshold:
            count_arr = []
            for i in range(0, n_rows):
                count = 0
                for j in range(0, n):
                    if data[i,j] > u:
                        count = count + 1
                count_arr.append(count)
            di.append(np.var(count_arr, ddof = 1)/np.mean(count_arr))
         
        v = n_rows - 1    
        di_conf_bound = [chi2.ppf(1 - self.alpha/2, v)/v,chi2.ppf(self.alpha/2, v)/v]
       
        plt.figure(1)
        plt.fill_between(threshold, di_conf_bound[0], di_conf_bound[1], alpha = 0.4, color = 'black', label = 'Chi-Square confidence bounds')
        plt.plot(threshold,di, color = 'black')
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title != None:
            plt.title(title)
        plt.show()

        return(threshold, di, di_conf_bound)
    
    def lmomplot(self, u_range):
        sample = self.data
        step = (u_range[1] - u_range[0])/30
        threshold = np.arange(u_range[0], u_range[1] + step, step=step)
        emp_lmom3 = []
        emp_lmom4 = []
        
        for u in threshold:
            lmons = tm2.samplm(sample, u)
            emp_lmom3.append(lmons[2])
            emp_lmom4.append(lmons[3])
        
        def t4(t3):
            return(t3*(1 + 5*t3)/(5 + t3))
        
        step_t3 = 0.01
        t3_arr = np.arange(min(emp_lmom3) - 0.1, max(emp_lmom3) + 0.1 + step_t3, step_t3) 
        t4_arr = []
        for t3 in t3_arr:
            t4_arr.append(t4(t3))
            
        plt.plot(t3_arr, t4_arr, color = 'black')
        plt.scatter(emp_lmom3, emp_lmom4, c = threshold, s = 20, cmap = 'cividis')
        plt.colorbar(label = 'Threshold Values')
        plt.xlabel('\u03C4\N{SUBSCRIPT THREE}')
        plt.ylabel('\u03C4\N{SUBSCRIPT FOUR}')   
        return(threshold, emp_lmom3, emp_lmom4, t3_arr, t4_arr)
            
            
    
def declust(data, u, time_cond, plot = True, **kwargs):
    n_obs = data.shape[0]
    clust = []
    time = []
    clust_max = []
    time_max = []
    for i in range(0, n_obs):
        if i==0:
            if data[i,1] > u:
                clust.append(data[i,1])
                time.append(data[i,0])
        if i>0 and i<n_obs-1:
            if data[i,1] > u:
                clust.append(data[i,1])
                time.append(data[i,0])
                if data[i+1,0] - data[i,0] > time_cond:
                    clust_max.append(max(clust))
                    time_max.append(time[clust.index(max(clust))])
                    time = []
                    clust = []
                elif data[i+1,1] < u:
                    clust_max.append(max(clust))
                    time_max.append(time[clust.index(max(clust))])
                    time = []
                    clust = []
        if i==n_obs-1:
            if data[i,1]>u:
                time.append(data[i,0])
                clust.append(data[i,1])
                clust_max.append(max(clust))
                time_max.append(time[clust.index(max(clust))])
                
    declust_data = np.array([time_max,clust_max]).transpose()
    
    if plot == True:
        if len(kwargs) == 0:
            xlabel = 'Time'
            ylabel = 'Variable value'
            
        else:
            args = list(kwargs)
            args_val = list(kwargs.values())
            for i in range(0, len(args)):
                if args[i] == 'xlabel':
                    xlabel = args_val[i]
                elif args[i] == 'ylabel':
                    ylabel = args_val[i]

        plt.figure(1)
        plt.scatter(data[:,0], data[:,1])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('Original sample')
        plt.show()
        
        plt.figure(2)
        plt.scatter(declust_data[:,0], declust_data[:,1])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('Declustered sample')
        plt.show()
        
   
    return(declust_data)

def samplm(data, u):
    
    sample = []
    for x in data:
        if x > u:
            sample.append(x)
    sample = np.sort(sample)
    
    n = len(sample)

    #first moment
    l1 = np.sum(sample) / sm.comb(n, 1, exact=True)
    
    #second moment
    comb1 = range(n)
    coefl2 = 0.5 / sm.comb(n, 2, exact=True)
    sum_xtrans = sum([(comb1[i] - comb1[n - i - 1]) * sample[i] for i in range(n)])
    l2 = coefl2 * sum_xtrans
    
    #third moment
    comb3 = [sm.comb(i, 2, exact=True) for i in range(n)]
    coefl3 = 1.0 / 3.0 / sm.comb(n, 3, exact=True)
    sum_xtrans = sum([(comb3[i] - 2 * comb1[i] * comb1[n - i - 1] + comb3[n - i - 1]) * sample[i] for i in range(n)])
    l3 = coefl3 * sum_xtrans / l2
    
    #fourth moment
    comb5 = [sm.comb(i, 3, exact=True) for i in range(n)]
    coefl4 = 0.25 / sm.comb(n, 4, exact=True)
    sum_xtrans = sum(
        [(comb5[i] - 3 * comb3[i] * comb1[n - i - 1] + 3 * comb1[i] * comb3[n - i - 1] - comb5[n - i - 1]) * sample[i]
         for i in range(n)])
    l4 = coefl4 * sum_xtrans / l2
    
    return(l1, l2, l3, l4)

def gpd2frech(x, model = None, **kwargs):
    if model != None:
        scale, shape = model.estimates()
        data, u = model.data()
        pat = model.pat()
        
    else:
        args = list(kwargs)
        args_val = list(kwargs.values())
        for i in range(0, len(args)):
            if args[i] == 'scale':
                scale = args_val[i]
            elif args[i] == 'shape':
                shape = args_val[i]
            elif args[i] == 'u' or args[i]=='threshold':
                u = args_val[i]
            elif args[i] == 'pat':
                pat = args_val[i]
                
    nparr = np.array([1])
    
    if type(x) == list or type(x) == type(nparr):
        z = []
        for data in x:
            if data >= u:
                z.append(-1/(np.log(1 - pat*((1 + shape*(data - u)/scale)**(-1/shape)))))
            else:
                z.append(np.nan)
                print('Value of x less than threshold.')
        return(z)
    
    else:
        if x >= u:
            z = -1/(np.log(1 - pat*((1 + shape*(x - u)/scale)**(-1/shape))))
            return(z)
        else:
            print('Value of x less than threshold.')
            return(np.nan)
            
class pickands_models(object):
    
    def logistic(w, alpha):
        A = (((1 - w)**(1/alpha)) + w**(1/alpha))**alpha
        return(A)
    
    def asy_logistic(w, theta1, theta2, alpha):
        A = (1 - theta1)*(1 - w) + (1 - theta2)*w + (((1 - w)*theta1)**(1/alpha) + (w*theta2)**(1/alpha))**alpha
        return(A)
    
    def neg_logistic(w, alpha):
        A = 1 - (((1 - w)**(-alpha)) + w**(-alpha))**(-1/alpha)
        return(A)
    
    def asy_neg_logistic(w, theta1, theta2, alpha):
        A = 1 - (((1-w)/theta1)**-alpha + (w/theta2)**-alpha)**(-1/alpha)
        return(A)
    
    def mixed(w, alpha):
        A = 1 - w*(1 - w)*alpha
        return(A)
    
    def asy_mixed(w, theta, alpha):
        A = theta*(w**3) + alpha*(w**2) - (alpha + theta)*w + 1
        return(A)
   
class build_bivarmodel(object):
    def __init__(self, data1, u1, data2, u2, model = 'log', alpha_sig = 0.05):
        self.model = model
        self.data1 = np.array(data1)
        self.data2 = np.array(data2)
        data_mat = np.array([data1,data2]).transpose()
        self.u = [u1, u2]
        nr,nc = data_mat.shape
        data_matr = ro.r.matrix(data_mat, nrow=nr, ncol=nc)
        ur = FloatVector(self.u)
        
        if model == 'log' or model == 'nlog' or model == 'mix':
            res = POTr.fitbvgpd(data_matr, ur, model = model)
            self.res_g = res
            self.scale1, self.shape1, self.scale2, self.shape2, self.alpha = res[0]
            self.Deviance = res[5][0]
            self.AIC = self.Deviance + 10
            self.prob_cond = res[21][0]
            self.marginals = [res[11][0], res[11][1]]
            self.marginals_prop = [res[12][0], res[12][1]]
            self.joint_ab = res[11][2]
            self.joint_nb_eve = res[11][3]
            self.joint_prop = res[12][2]
            self.estimator = res[17][0]
            self.f_eval, self.g_eval = res[8]
            self.convergence = res[7][0]
            if type(res[1]) == rpy2.robjects.vectors.FloatVector:
                self.std_scale1, self.std_shape1, self.std_scale2, self.std_shape2, self.std_alpha = res[1]
                self.var_cov = np.array(res[2]).reshape((5,5))
                z = norm.ppf(1 - alpha_sig/2)
                self.scale1_variation = [self.scale1 - z * self.std_scale1, self.scale1 + z * self.std_scale1] 
                self.shape1_variation = [self.shape1 - z * self.std_shape1, self.shape1 + z * self.std_shape1] 
                self.scale2_variation = [self.scale2 - z * self.std_scale2, self.scale2 + z * self.std_scale2] 
                self.shape2_variation = [self.shape2 - z * self.std_shape2, self.shape2 + z * self.std_shape2] 
                self.alpha_variation  = [self.alpha - z * self.std_alpha, self.alpha + z * self.std_alpha] 
        
        elif model == 'alog' or model == 'anlog':
            res = POTr.fitbvgpd(data_matr, ur, model = model)
            self.res_g = res
            self.scale1, self.shape1, self.scale2, self.shape2, self.alpha, self.ascoef1, self.ascoef2 = res[0]
            self.Deviance = res[5][0]
            self.AIC = self.Deviance + 4
            self.prob_cond = res[21][0]
            self.marginals = [res[11][0], res[11][1]]
            self.marginals_prop = [res[12][0], res[12][1]]
            self.joint_ab = res[11][2]
            self.joint_nb_eve = res[11][3]
            self.joint_prop = res[12][2]
            self.estimator = res[17][0]
            self.f_eval, self.g_eval = res[8]
            self.convergence = res[7][0]
            if type(res[1]) == rpy2.robjects.vectors.FloatVector:
                self.std_scale1, self.std_shape1, self.std_scale2, self.std_shape2, self.std_alpha, self.std_ascoef1, self.std_ascoef2 = res[1]
                self.var_cov = np.array(res[2]).reshape((7,7))
                z = norm.ppf(1 - alpha_sig/2)
                self.scale1_variation = [self.scale1 - z * self.std_scale1, self.scale1 + z * self.std_scale1] 
                self.shape1_variation = [self.shape1 - z * self.std_shape1, self.shape1 + z * self.std_shape1] 
                self.scale2_variation = [self.scale2 - z * self.std_scale2, self.scale2 + z * self.std_scale2] 
                self.shape2_variation = [self.shape2 - z * self.std_shape2, self.shape2 + z * self.std_shape2] 
                self.alpha_variation  = [self.alpha - z * self.std_alpha, self.alpha + z * self.std_alpha]            
                self.ascoef1_variation  = [self.ascoef1 - z * self.std_ascoef1, self.ascoef1 + z * self.ascoef1]
                self.ascoef2_variation  = [self.ascoef2 - z * self.std_ascoef2, self.ascoef2 + z * self.ascoef2]
         
        elif model == 'amix':
            res = POTr.fitbvgpd(data_matr, ur, model = model)
            self.res_g = res
            self.scale1, self.shape1, self.scale2, self.shape2, self.alpha, self.ascoef1 = res[0]
            self.Deviance = res[5][0]
            self.AIC = self.Deviance + 12
            self.prob_cond = res[21][0]
            self.marginals = [res[11][0], res[11][1]]
            self.marginals_prop = [res[12][0], res[12][1]]
            self.joint_ab = res[11][2]
            self.joint_nb_eve = res[11][3]
            self.joint_prop = res[12][2]
            self.estimator = res[17][0]
            self.f_eval, self.g_eval = res[8]
            self.convergence = res[7][0]
            if type(res[1]) == rpy2.robjects.vectors.FloatVector:
                self.std_scale1, self.std_shape1, self.std_scale2, self.std_shape2, self.std_alpha, self.std_ascoef1 = res[1]
                self.var_cov = np.array(res[2]).reshape((6,6))
                z = norm.ppf(1 - alpha_sig/2)
                self.scale1_variation = [self.scale1 - z * self.std_scale1, self.scale1 + z * self.std_scale1] 
                self.shape1_variation = [self.shape1 - z * self.std_shape1, self.shape1 + z * self.std_shape1] 
                self.scale2_variation = [self.scale2 - z * self.std_scale2, self.scale2 + z * self.std_scale2] 
                self.shape2_variation = [self.shape2 - z * self.std_shape2, self.shape2 + z * self.std_shape2] 
                self.alpha_variation  = [self.alpha - z * self.std_alpha, self.alpha + z * self.std_alpha]            
                self.ascoef1_variation  = [self.ascoef1 - z * self.std_ascoef1, self.ascoef1 + z * self.ascoef1]
            
    def disp(self):
        
        if self.model == 'log' or self.model == 'nlog' or self.model == 'mix': 

            print('\nModel Estimatives and Parameters\n')
            print('--------------')
            print('Marginal Thresholds: {}\nMaginal Number above: {}\nMarginal Proportion Above: {}\nJoint number above: {}\nJoint proportion above: {}'.format(self.u, self.marginals, self.marginals_prop, self.joint_ab, self.joint_prop))
            print('Number of events such as (Y1 > u1) U (Y2 > u2): {}'.format(self.joint_nb_eve))
            print('lim_u Pr[ X_1 > u | X_2 > u] = {:.4f}'.format(self.prob_cond))
            print('--------------')
            if type(self.res_g[1]) == rpy2.robjects.vectors.FloatVector:
                print('Model:',self.model,'\n\nEstimates\nScale parameter 1: {:.4f} [{:.4f},{:.4f}]\nShape parameter 1: {:.4f} [{:.4f},{:.4f}]\nScale parameter 2: {:.4f} [{:.4f},{:.4f}]\nShape parameter 2: {:.4f} [{:.4f},{:.4f}]\nAlpha: {:.4f} [{:.4f},{:.4f}]'.format(self.scale1, self.scale1_variation[0], self.scale1_variation[1], self.shape1, self.shape1_variation[0], self.shape1_variation[1], self.scale2, self.scale2_variation[0], self.scale2_variation[1], self.shape2, self.shape2_variation[0], self.shape2_variation[1], self.alpha, self.alpha_variation[0], self.alpha_variation[1]))
                print('\nStandard errors\nScale parameter 1: {:.4f} Shape parameter 1: {:.4f}\nScale parameter 2: {:.4f} Shape parameter 2: {:.4f}\nalpha: {:.4f}'.format(self.std_scale1, self.std_shape1, self.std_scale2, self.std_shape2, self.std_alpha))
            else:
                print('Model:',self.model,'\n\nEstimates\nScale parameter 1: {:.4f}\nShape parameter 1: {:.4f}\nScale parameter 2: {:.4f}\nShape parameter 2: {:.4f}'.format(self.scale1, self.shape1, self.scale2, self.shape2, self.alpha))
                print('\nStandard errors were not computed.')
            print('--------------')
            print('Metrics')
            print('\nDeviance: {}'.format(self.Deviance))
            print('AIC: {}'.format(self.AIC))
            print('--------------')
            print('Optimization Information')
            print('\nConvergence:',self.convergence)
            print('Function Evaluations: {}'.format(self.f_eval))
            print('Gradient Evaluations: {}'.format(self.g_eval))        
        
        elif self.model == 'alog' or self.model == 'anlog':

            print('\nModel Estimatives and Parameters\n')
            print('--------------')
            print('Marginal Thresholds: {}\nMaginal Number above: {}\nMarginal Proportion Above: {}\nJoint number above: {}\nJoint proportion above: {}'.format(self.u, self.marginals, self.marginals_prop, self.joint_ab, self.joint_prop))
            print('Number of events such as (Y1 > u1) U (Y2 > u2): {}'.format(self.joint_nb_eve))
            print('lim_u Pr[ X_1 > u | X_2 > u] = {:.4f}'.format(self.prob_cond))
            print('--------------')
            if type(self.res_g[1]) == rpy2.robjects.vectors.FloatVector:
                print('Model:',self.model,'\n\nEstimates\nScale parameter 1: {:.4f} [{:.4f},{:.4f}]\nShape parameter 1: {:.4f} [{:.4f},{:.4f}]\nScale parameter 2: {:.4f} [{:.4f},{:.4f}]\nShape parameter 2: {:.4f} [{:.4f},{:.4f}]\nAlpha: {:.4f} [{:.4f},{:.4f}]\nCoef1: {:.4f} [{:.4f},{:.4f}]\nCoef2: {:.4f} [{:.4f},{:.4f}]'.format(self.scale1, self.scale1_variation[0], self.scale1_variation[1], self.shape1, self.shape1_variation[0], self.shape1_variation[1], self.scale2, self.scale2_variation[0], self.scale2_variation[1], self.shape2, self.shape2_variation[0], self.shape2_variation[1], self.alpha, self.alpha_variation[0], self.alpha_variation[1], self.ascoef1, self.ascoef1_variation[0], self.ascoef1_variation[1], self.ascoef2, self.ascoef2_variation[0], self.ascoef2_variation[1]))
                print('\nStandard errors\nScale parameter 1: {:.4f} Shape parameter 1: {:.4f}\nScale parameter 2: {:.4f} Shape parameter 2: {:.4f}\nalpha: {:.4f}\nCoef1: {:.4f}\nCoef2: {:.4f}'.format(self.std_scale1, self.std_shape1, self.std_scale2, self.std_shape2, self.std_alpha, self.std_ascoef1, self.std_ascoef2))
            else:
                print('Model:',self.model,'\n\nEstimates\nScale parameter 1: {:.4f}\nShape parameter 1: {:.4f}\nScale parameter 2: {:.4f}\nShape parameter 2: {:.4f}\nAlpha: {:.4f}\nCoef1: {:.4f}\nCoef2: {:.4f}'.format(self.scale1, self.shape1, self.scale2, self.shape2, self.alpha, self.ascoef1, self.ascoef2))
                print('\nStandard errors were not computed.')
            print('--------------')
            print('Metrics')
            print('\nDeviance: {}'.format(self.Deviance))
            print('AIC: {}'.format(self.AIC))
            print('--------------')
            print('Optimization Information')
            print('\nConvergence:',self.convergence)
            print('Function Evaluations: {}'.format(self.f_eval))
            print('Gradient Evaluations: {}'.format(self.g_eval))
        
        elif self.model == 'amix':
        
            print('\nModel Estimatives and Parameters\n')
            print('--------------')
            print('Marginal Thresholds: {}\nMaginal Number above: {}\nMarginal Proportion Above: {}\nJoint number above: {}\nJoint proportion above: {}'.format(self.u, self.marginals, self.marginals_prop, self.joint_ab, self.joint_prop))
            print('Number of events such as (Y1 > u1) U (Y2 > u2): {}'.format(self.joint_nb_eve))
            print('lim_u Pr[ X_1 > u | X_2 > u] = {:.4f}'.format(self.prob_cond))
            print('--------------')
            if type(self.res_g[1]) == rpy2.robjects.vectors.FloatVector:
                print('Model:',self.model,'\n\nEstimates\nScale parameter 1: {:.4f} [{:.4f},{:.4f}]\nShape parameter 1: {:.4f} [{:.4f},{:.4f}]\nScale parameter 2: {:.4f} [{:.4f},{:.4f}]\nShape parameter 2: {:.4f} [{:.4f},{:.4f}]\nAlpha: {:.4f} [{:.4f},{:.4f}]\nCoef1: {:.4f} [{:.4f},{:.4f}]'.format(self.scale1, self.scale1_variation[0], self.scale1_variation[1], self.shape1, self.shape1_variation[0], self.shape1_variation[1], self.scale2, self.scale2_variation[0], self.scale2_variation[1], self.shape2, self.shape2_variation[0], self.shape2_variation[1], self.alpha, self.alpha_variation[0], self.alpha_variation[1], self.ascoef1, self.ascoef1_variation[0], self.ascoef1_variation[1]))
                print('\nStandard errors\nScale parameter 1: {:.4f} Shape parameter 1: {:.4f}\nScale parameter 2: {:.4f} Shape parameter 2: {:.4f}\nalpha: {:.4f}\nCoef1: {:.4f}'.format(self.std_scale1, self.std_shape1, self.std_scale2, self.std_shape2, self.std_alpha, self.std_ascoef1))
            else:
                print('Model:',self.model,'\n\nEstimates\nScale parameter 1: {:.4f}\nShape parameter 1: {:.4f}\nScale parameter 2: {:.4f}\nShape parameter 2: {:.4f}\nAlpha: {:.4f}\nCoef1: {:.4f}'.format(self.scale1, self.shape1, self.scale2, self.shape2, self.alpha, self.ascoef1))
                print('\nStandard errors were not computed.')
            print('--------------')
            print('Metrics')
            print('\nDeviance: {}'.format(self.Deviance))
            print('AIC: {}'.format(self.AIC))
            print('--------------')
            print('Optimization Information')
            print('\nConvergence:',self.convergence)
            print('Function Evaluations: {}'.format(self.f_eval))
            print('Gradient Evaluations: {}'.format(self.g_eval))
            
    def estimates(self):
        if self.model == 'log' or self.model == 'nlog' or self.model == 'mix':
            return(self.scale1, self.shape1, self.scale2, self.shape2, self.alpha)
        elif self.model == 'alog' or self.model == 'anlog':
            return(self.scale1, self.shape1, self.scale2, self.shape2, self.alpha, self.ascoef1, self.ascoef2)
        elif self.model == 'amix':
            return(self.scale1, self.shape1, self.scale2, self.shape2, self.alpha, self.ascoef1)
       
    def var_cov(self):
        if type(self.res_g[1]) == rpy2.robjects.vectors.FloatVector:
            return(self.var_cov)
        else:
            print('The variance-covariance matrix can not be computed.')
            return(np.nan)
        
    def std_errors(self):
        if type(self.res_g[1]) == rpy2.robjects.vectors.FloatVector:
            if self.model == 'log' or self.model == 'nlog' or self.model == 'mix':
                return(self.std_scale1, self.std_shape1, self.std_scale2, self.std_shape2, self.std_alpha)
            elif self.model == 'alog' or self.model == 'anlog':
                return(self.std_scale1, self.std_shape1, self.std_scale2, self.std_shape2, self.std_alpha, self.std_ascoef1, self.std_ascoef2)
            elif self.model == 'amix':
                return(self.std_scale1, self.std_shape1, self.std_scale2, self.std_shape2, self.std_alpha, self.std_ascoef1)  
        else:
            print('The variance-covariance matrix can not be computed.')
            return(np.nan)
    
    def data(self):
        return(self.data1, self.data2, self.u)
    
    def metrics(self):
        return(self.Deviance, self.AIC)
    
    def threshold_info(self):
        return(self.marginals, self.marginals_prop, self.joint_ab, self.joint_nb_eve, self.joint_prop, self.prob_cond)
    
            