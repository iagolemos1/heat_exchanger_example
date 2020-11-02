# MIT License

# Copyright (c) 2020 Iago Pereira Lemos

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
import math as mt
import matplotlib.pyplot as plt
from scipy.stats import mode
import scipy.stats as sst
import scipy
from pyDOE import lhs
import ghalton
import sobol_seq
from sympy.solvers import solve
from sympy import Symbol
import numdifftools as nd



class Sample(object):
    """
    Parameters
    ----------
    *sample : List or array with observed values.
    
    Returns
    -----------
    Sample_Stats object with the following methods:
    
    mean, var (variance), std (standard deviation), skewness, kurtosis, 
    cov (coefficient of variation), cdf, hist, mode, median
    
    
    """
    def __init__(self, *sample):

        self.x = sample[0]
        
    def mean(self):
        """
        Returns 
        -------
        E_x: Mean of the sample
        """        
        x = self.x
        E_x = np.mean(x)
        return(E_x)
    
    def var(self):
        """
        Returns 
        -------
        Var_x: Unbiased variance of the sample
        """
        x = self.x
        Var_x = np.var(x, ddof = 1)
        return(Var_x)
        
    def std(self):
        """
        Returns 
        -------
        Std_x: Unbiased standard deviation of the sample
        """
        x = self.x
        Std_x = np.std(x, ddof = 1)
        return(Std_x)
        
    def skewness(self):
        """
        Returns 
        -------
        sk:  Skewness coefficient.
        """
        x = self.x
        E_x = Sample.mean(self)
        Var_x = Sample.var(self)
        sk =  np.sum((x - E_x)**3)/(len(x) * Var_x**1.5)
        return(sk)
    
    def kurtosis(self):
        """
        Returns 
        ---------
        kurt: Kurtosis coefficient.
        """
        x = self.x
        E_x = Sample.mean(self)
        Var_x = Sample.var(self)
        kurt =  np.sum((x - E_x)**4)/(len(x) * Var_x**2) - 3
        return(kurt)       
    
    def cov(self):
        """
        Returns the coefficient of variation of the variable of analysis
        """
        E_x = Sample.mean(self)
        Std_x = Sample.std(self)
        cov = Std_x/E_x
        return(cov)
    
    def cdf(self, alpha): #Plot empirical cfd with confidence interval
        """
        Plots the empirical cdf with confidence interval based on the DKW method.
        Parameters
        ----------
        alpha: Significance level 
        
        Returns
        -------
        y : Array of the empirical probability of each observed value.
        """
        x = self.x
        n = len(x)
        y = np.arange(1, n+1)/n
        
        #Computing confidence interval with the Dvoretzky–Kiefer–Wolfowitz method based on the empirical points
        F1 = []
        F2 = []
        for i in range(0, n):
            e = (((mt.log(2/alpha))/(2*n))**0.5)  
            F1.append(y[i] - e)
            F2.append(y[i] + e) 
        plt.plot(sorted(x), y, label='Empirical CDF')
        plt.plot(sorted(x), F1, linestyle='--', color='red', alpha = 0.8, lw = 0.9, label = 'Dvoretzky–Kiefer–Wolfowitz Confidence Bands')
        plt.plot(sorted(x), F2, linestyle='--', color='red', alpha = 0.8, lw = 0.9)
        plt.ylabel('Cumulative Distribution Function')
        plt.xlabel('Observed Data')
        plt.legend()
        plt.show()
        
        return(y)
    
    def hist(self, bins):
        """
        Plots the histogram of the variable of analaysis

        Parameters
        ----------
        bins : Number of bins of the histogram. It may be 'auto', 'fd', 'doane',
        'scott', 'stone', 'rice', 'sturges' or 'sqrt' rules.
        """
        x = self.x
        plt.hist(x, bins)
        plt.xlabel('Observed Data')
        plt.ylabel('Frequency')
        plt.show()      
        
    def mode(self):
        """
        Returns
        -------
        mode_val : Mode value from the sample.
        """
        x = self.x
        mode_val = mode(x)
        return mode_val
    
    def median(self):
        """
        Returns
        -------
        median : Median value from the sample.
        """
        x = self.x
        median = np.median(x)
        return median
    
    def ApEn(self, m, r) -> float:
        """
        Computes the approximate entropy of a time series data.
        
        Parameters
        ----------
        m :length of compared run of data
        
        r :filtering level
        
        Return
        -------
        ApEn: Approximate entropy
        """
        U = self.x
        def _maxdist(x_i, x_j):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
        
        def _phi(m):
            x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
            C = [
                len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
                for x_i in x
                ]
            return (N - m + 1.0) ** (-1) * sum(np.log(C))
        
        N = len(U)
        ApEn = _phi(m + 1) - _phi(m)
        return abs(ApEn)
        
    def SampEn(self, m, r):
        """
        Computes the sample entropy of a time series data.
        
        Parameters
        ----------
        m :length of compared run of data
        
        r :filtering level
        
        Return
        -------
        SampEn: Approximate entropy
        """            
        L = self.x
        N = len(L)
        B = 0.0
        A = 0.0
 
        xmi = np.array([L[i : i + m] for i in range(N - m)])
        xmj = np.array([L[i : i + m] for i in range(N - m + 1)])
        
        B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])
        
        m += 1
        xm = np.array([L[i : i + m] for i in range(N - m + 1)])
        
        A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])
        
        SampEn = -np.log(A / B)
        return(SampEn)
    
    def entropy(self, b):
        
        """
        Computes the entropy of a sample (see https://en.wikipedia.org/wiki/Entropy_(information_theory))
        
        Parameters
        ----------
        b: base of the log, it can be any value, but common values are 2, e 
        (euler's number) and 10.
        b=2  -> bits
        b=e  -> nats
        b=10 -> bans
        
        Returns
        -------
        H: Entropy value for the sample
        """
        x = self.x
 
        su=0
     
        for p in x:
            r= p/sum(x)
            if r==0:
                su+=0
            else:
                su+= -r*(np.log(r))
            H = su/np.log(b) 
        return(H)
           
def makedist(dist_type, *pars, **kwards):
    
    """
    Creates a distribution class from scipy continuous distributions
    See https://docs.scipy.org/doc/scipy/reference/stats.html.

    Parameters
    ----------
    dist_type: String -> Type of the distribution (see the scipy documentation)
    *pars and **kwards: Statistical parameters and its values

    Return
    ------
    dist: Distribution class
    """
    a = 'sst.'
    b = dist_type
    c = a + b
    Scipy_stats_Obj = eval(c)
    dist = Scipy_stats_Obj(*pars, **kwards)
    
    return(dist, dist_type)

class mc_design(object):
    """
    Creates a Monte-Carlo type object (which is necessary for Monte-Carlo Analysis)
    
    Parameter
    ---------
    dist: Distribution object (makedist) of the random variates
    
    Methods
    -------
    mc_sampling: Return a new sample, with a given number of points
    
    mc_convergence: Return three plots with the analysis of the error in estmating the mean
    """
    def __init__(self, dist):
        self.dist = dist[0]
        
    def gen_samp(self, N, random_state = None):
        """
        Parameters
        ----------
        N: Number of points to simulate
        
        random_state: Optional. The default is None.

        Returns
        -------
        sample: New simulated data

        """
        sample = self.dist.rvs(N, random_state = random_state)
        return(sample)
    
    def mc_convergence(self, N, bin_method, random_state = None):
        """
        It returns 3 plots:
            
        *Comparing the number of samples and standard error of the mean;
        
        *Distribution of the error of each sample in esmating the population mean;
        
        *Convergence of the error of each sample in esmating the population mean.
        
        Parameters
        ----------
        N: Array with the points of analysis
        
        bin_method: The method to draw the histrogram of the error of the estimates.
        
        random_state: Optional. The default is None.

        """
        mc_error = []
        std_arr = []
        for N_point in N:
            samp = Monte_Carlo.mc_sampling(self, N_point)
            std_arr.append(self.dist.std()/(N_point**0.5))
            mc_error.append(np.mean(samp) - self.dist.mean())
            
                
        std_mean_error = np.mean(std_arr)
        print(std_mean_error)
        mc_error_norm = sst.norm(loc = 0, scale = std_mean_error)
        
        x = np.arange(-0.2, 0.2, 0.001)
        pdf = mc_error_norm.pdf(x)
        
        plt.figure(1)
        plt.scatter(N, std_arr)
        plt.xlabel('Number of samples')
        plt.ylabel('Standard Error')
        plt.show()
        
        plt.figure(2)
        plt.plot(x, pdf)
        plt.hist(mc_error, bins = bin_method, density = True)
        plt.xlabel('Error')
        plt.ylabel('Density')
        
        plt.figure(3)
        plt.plot(N, mc_error)
        plt.xlabel('Number of samples')
        plt.ylabel('Error')
        
        plt.show()
              
class strat_design(object):
    """
    Creates a stratfied-sampling type object (which is necessary for stratfied-sampling analysis)
    
    Parameter
    ---------
    dist: Distribution object (makedist) of the random variates
    
    Methods
    -------
    strat: Return a new sample, with a given number of points
    
    strat_convergence: Return a plot of the standard error convergence
    
    """
    def __init__(self, dist):
        self.dist = dist[0]
        
    def gen_samp(self, N, M):
        """
        It return the new based-design simulated sample

        Parameters
        ----------
        N : Number of samples to simulate and analyse
        
        M : Number of stratums (>1)

        Returns
        -------
        sample: Simulated sample

        """
        Sm = np.empty((M, 2))
        count = 0
        for i in range(1, M+1, 1):
            Sm[count, 0] = (i-1)/M
            Sm[count, 1] = i/M
            count = count + 1
            
        Ns = int(N/M)
        
        uni_rand = np.empty((M, Ns))
        for i in range(0, M):
            uni_rand[i,:] = np.random.uniform(Sm[i,0], Sm[i,1], Ns)
            
        uni_rand = uni_rand.reshape((1, -1))
        
        sample = self.dist.ppf(uni_rand)
        
        return(sample)
    
    def strat_convergence(self, N, M):
        """
        It returns 1 plot:
        *Convergence of the standard error
        
        Parameters
        ----------
        N: Array with the number of samples to simulate and analyse
        
        M: Number of stratuns (>1)

        """
        Sm = np.empty((M, 2))
        count = 0
        for i in range(1, M+1, 1):
            Sm[count, 0] = (i-1)/M
            Sm[count, 1] = i/M
            count = count + 1
        std = []
        for N_i in N:
            Ns = int(N_i/M)
            
            uni_rand = np.empty((M, Ns))
            for i in range(0, M):
                uni_rand[i,:] = np.random.uniform(Sm[i,0], Sm[i,1], Ns)
                
            rand_samp = np.empty(uni_rand.shape)
            for i in range (0, M):
                rand_samp[i,:] = self.dist.ppf(uni_rand[i,:])
            
            weight = Ns/N_i
            std_strat = np.empty((M, 1))
            
            for i in range(0, M):
                std_strat[i,0] = np.std(rand_samp[i,:], ddof = 1)
              
            arr_var = np.empty((M, 1))
            arr_var[:,0] = ((weight**2)/Ns)*std_strat[:,0]
            var = np.sum(arr_var)
            std.append(var**0.5)
            
        plt.scatter(N, std)
        plt.xlabel('Standard error')
        plt.ylabel('Number of samples')
        plt.show()
   
class lh_design(object):
    """
    Creates a Latin hypercube class for design sampling
    
    Parameter
    ---------
    dist: Distribution object (makedist) of the random variates
    
    Method
    ------
    gen_samp: Returns the generated sample
    """
    def __init__(self, dist):
        self.dist = dist[0]
        
    def gen_samp(self, factor, samples=None, criterion=None, iterations=None, vectorized=True):
        """
        Parameters
        ----------
        factor : number of factors.
        
        samples : number of points to be generated for each factor
        
        criterion : a string that tells the algorithm how to sample the points;
        default is None, which tells to simply randomizes the points within the intervals. 
        It may be, also:
        “center” or “c”: center the points within the sampling intervals
            
        “maximin” or “m”: maximize the minimum distance between points, but place the point in a randomized location within its interval
            
        “centermaximin” or “cm”: same as “maximin”, but centered within the intervals
            
        “correlation” or “corr”: minimize the maximum correlation coefficient
            
        iterations : Number of iterations. Default is None
        
        vectorized : Boolean. Default is True. If false, the points will be returned as a 
        matrix, factor x samples.

        Returns
        -------
        samp: The generated sample.

        """
        dist = self.dist
        lhd = lhs(factor, samples=samples, criterion=criterion, iterations=iterations)
        samp = dist.ppf(lhd)
        if vectorized == False:
            return(samp)
        
        else:
            samp = samp.reshape(1, -1)
            return(samp)
    
class qmc_design(object):
    """
    Creates a Quasi Monte-Carlo class for design sampling by using a low-discrepancy
    sequence.
    
    Parameter
    ---------
    dist: Distribution object (makedist) of the random variates
    
    Method
    ------
    gen_samp_sobol: Returns the generated sample using the Sobol Sequence
    
    gen_samp_halton: Returns the generated sample using the Halton Sequence
    
    gen_samp_corput: Returns the genarated sample using the van der Corput sequence
    
    gen_samp_hammersley: Returns the generated sample using the Hammerslay sequence
    """
    def __init__(self, dist):
        self.dist = dist[0]
        
    def gen_samp_sobol(self, dim, N, skip=0, vectorized = True):
        """
        Generates a quasi monte-carlo sample using the Sobol Sequence.
        Parameters
        ----------
        dim : Spatial dimension
        
        N : Number of points to generate
        
        skip : The number of initial points do skip in the Sobol Sequence Algorithm. The default is 0.
        
        vectorized : The default is True. If it's false, the new sample will be a dim x N array.

        Returns
        -------
        Samp: New generated sample.
        """
        dist = self.dist
        sobol_quasimc = sobol_seq.i4_sobol_generate(dim, N, skip)
        samp = dist.ppf(sobol_quasimc)
        
        if vectorized == False:
            return(samp)
        else:
            samp = samp.reshape(1, -1)
            return(samp)
        
    def gen_samp_halton(self, dim, N, vectorized = True):
        """
        Generates a quasi monte-carlo sample using the Halton Sequence.
        
        Parameters
        ----------
        dim : Spatial dimension
        
        N : Number of points to generate
        
        Returns
        -------
        Samp: New generated sample.
        """
        
        dist = self.dist
        halton_des = (np.array(ghalton.Halton(N).get(dim)))
        samp = dist.ppf(halton_des)
        
        if vectorized == False:
            return(samp)
        else:
            samp = samp.reshape(1, -1)
            return(samp)
        
    def gen_samp_corput(self, i1, i2):
        """
        Generates a quasi monte-carlo sample using the van der Corput sequence, 
        generating the sequence between two indexes. 
        Parameters
        ----------
        i1: First element index of the van der Corput sequence
        
        i2: Last element index of the van der Corput sequence
        -------
        Samp: New generated sample.
        """
        dist = self.dist
        
        def corput_sequence ( i1, i2 ):
     
          n = abs ( i2 - i1 ) + 1
          r = np.zeros ( n )
        
          if ( i1 <= i2 ):
            i3 = +1
          else:
            i3 = -1
        
          j = 0
      
          for i in range ( i1, i2 + i3, i3 ):
      
            if ( i < 0 ):
              s = -1
            else:
              s = +1
        
            t = abs ( int ( i ) )
     
            base_inv = 0.5
        
            r[j] = 0.0
        
            while ( t != 0 ):
              d = ( t % 2 )
              r[j] = r[j] + d * base_inv
              base_inv = base_inv / 2.0
              t = ( t // 2 )
   
            r[j] = r[j] * s
        
            j = j + 1   
        
          return r        
        
        corput_seq = corput_sequence(i1, i2)
        
        samp = dist.ppf(corput_seq)
        return(samp.reshape(1, -1))
        
        
    def gen_samp_hammersley(self, i1, i2, m, n, vectorized = True):
        """
        Generates a quasi monte-carlo sample using the Hammersley sequence, 
        generating the sequence between two indexes. 
        
        Parameters
        ----------
        i1: First element index of the Hammersley sequence
        
        i2: Last element index of the Hammersley sequence
        
        m: Spatial Dimension
        
        n: The base for the first component
        -------
        Samp: New generated sample.
        """
        dist = self.dist
        
        def hammersley_sequence (i1, i2, m, n):
        
        
          if ( i1 <= i2 ):
            i3 = +1
          else:
            i3 = -1
        
          l = abs ( i2 - i1 ) + 1
          r = np.zeros ( [ m, l ] )
          k = 0
        
          for i in range ( i1, i2 + i3, i3 ):
        
            t = np.ones ( m - 1 )
        
            t = t * i
        #
        #  Carry out the computation.
        #
            prime_inv = np.zeros ( m - 1 )
            for j in range ( 0, m - 1 ):
              prime_inv[j] = 1.0 / prime ( j )
         
            r[0,k] = float ( i % ( n + 1 ) ) / float ( n )
        
            while ( 0 < np.sum ( t ) ):
              for j in range ( 0, m - 1 ):
                d = ( t[j] % prime ( j ) )
                r[j+1,k] = r[j+1,k] + float ( d ) * prime_inv[j]
                prime_inv[j] = prime_inv[j] / prime ( j )
                t[j] = ( t[j] // prime ( j ) )
        
            k = k + 1
        
          return r
        
        def prime (n):
            
          from sys import exit
        
          prime_max = 1600
        
          prime_vector = np.array ( [
                2,    3,    5,    7,   11,   13,   17,   19,   23,   29, \
               31,   37,   41,   43,   47,   53,   59,   61,   67,   71, \
               73,   79,   83,   89,   97,  101,  103,  107,  109,  113, \
              127,  131,  137,  139,  149,  151,  157,  163,  167,  173, \
              179,  181,  191,  193,  197,  199,  211,  223,  227,  229, \
              233,  239,  241,  251,  257,  263,  269,  271,  277,  281, \
              283,  293,  307,  311,  313,  317,  331,  337,  347,  349, \
              353,  359,  367,  373,  379,  383,  389,  397,  401,  409, \
              419,  421,  431,  433,  439,  443,  449,  457,  461,  463, \
              467,  479,  487,  491,  499,  503,  509,  521,  523,  541, \
              547,  557,  563,  569,  571,  577,  587,  593,  599,  601, \
              607,  613,  617,  619,  631,  641,  643,  647,  653,  659, \
              661,  673,  677,  683,  691,  701,  709,  719,  727,  733, \
              739,  743,  751,  757,  761,  769,  773,  787,  797,  809, \
              811,  821,  823,  827,  829,  839,  853,  857,  859,  863, \
              877,  881,  883,  887,  907,  911,  919,  929,  937,  941, \
              947,  953,  967,  971,  977,  983,  991,  997, 1009, 1013, \
             1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, \
             1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, \
             1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, \
             1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, \
             1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, \
             1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, \
             1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, \
             1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, \
             1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, \
             1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, \
             1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, \
             1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, \
             1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, \
             1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, \
             2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, \
             2131, 2137, 2141, 2143, 2153, 2161, 2179, 2203, 2207, 2213, \
             2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287, \
             2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, \
             2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, \
             2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, \
             2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617, \
             2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, \
             2689, 2693, 2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741, \
             2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819, \
             2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903, \
             2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, \
             3001, 3011, 3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079, \
             3083, 3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, \
             3187, 3191, 3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257, \
             3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331, \
             3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, \
             3433, 3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, \
             3517, 3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571, \
             3581, 3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637, 3643, \
             3659, 3671, 3673, 3677, 3691, 3697, 3701, 3709, 3719, 3727, \
             3733, 3739, 3761, 3767, 3769, 3779, 3793, 3797, 3803, 3821, \
             3823, 3833, 3847, 3851, 3853, 3863, 3877, 3881, 3889, 3907, \
             3911, 3917, 3919, 3923, 3929, 3931, 3943, 3947, 3967, 3989, \
             4001, 4003, 4007, 4013, 4019, 4021, 4027, 4049, 4051, 4057, \
             4073, 4079, 4091, 4093, 4099, 4111, 4127, 4129, 4133, 4139, \
             4153, 4157, 4159, 4177, 4201, 4211, 4217, 4219, 4229, 4231, \
             4241, 4243, 4253, 4259, 4261, 4271, 4273, 4283, 4289, 4297, \
             4327, 4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409, \
             4421, 4423, 4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493, \
             4507, 4513, 4517, 4519, 4523, 4547, 4549, 4561, 4567, 4583, \
             4591, 4597, 4603, 4621, 4637, 4639, 4643, 4649, 4651, 4657, \
             4663, 4673, 4679, 4691, 4703, 4721, 4723, 4729, 4733, 4751, \
             4759, 4783, 4787, 4789, 4793, 4799, 4801, 4813, 4817, 4831, \
             4861, 4871, 4877, 4889, 4903, 4909, 4919, 4931, 4933, 4937, \
             4943, 4951, 4957, 4967, 4969, 4973, 4987, 4993, 4999, 5003, \
             5009, 5011, 5021, 5023, 5039, 5051, 5059, 5077, 5081, 5087, \
             5099, 5101, 5107, 5113, 5119, 5147, 5153, 5167, 5171, 5179, \
             5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273, 5279, \
             5281, 5297, 5303, 5309, 5323, 5333, 5347, 5351, 5381, 5387, \
             5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443, \
             5449, 5471, 5477, 5479, 5483, 5501, 5503, 5507, 5519, 5521, \
             5527, 5531, 5557, 5563, 5569, 5573, 5581, 5591, 5623, 5639, \
             5641, 5647, 5651, 5653, 5657, 5659, 5669, 5683, 5689, 5693, \
             5701, 5711, 5717, 5737, 5741, 5743, 5749, 5779, 5783, 5791, \
             5801, 5807, 5813, 5821, 5827, 5839, 5843, 5849, 5851, 5857, \
             5861, 5867, 5869, 5879, 5881, 5897, 5903, 5923, 5927, 5939, \
             5953, 5981, 5987, 6007, 6011, 6029, 6037, 6043, 6047, 6053, \
             6067, 6073, 6079, 6089, 6091, 6101, 6113, 6121, 6131, 6133, \
             6143, 6151, 6163, 6173, 6197, 6199, 6203, 6211, 6217, 6221, \
             6229, 6247, 6257, 6263, 6269, 6271, 6277, 6287, 6299, 6301, \
             6311, 6317, 6323, 6329, 6337, 6343, 6353, 6359, 6361, 6367, \
             6373, 6379, 6389, 6397, 6421, 6427, 6449, 6451, 6469, 6473, \
             6481, 6491, 6521, 6529, 6547, 6551, 6553, 6563, 6569, 6571, \
             6577, 6581, 6599, 6607, 6619, 6637, 6653, 6659, 6661, 6673, \
             6679, 6689, 6691, 6701, 6703, 6709, 6719, 6733, 6737, 6761, \
             6763, 6779, 6781, 6791, 6793, 6803, 6823, 6827, 6829, 6833, \
             6841, 6857, 6863, 6869, 6871, 6883, 6899, 6907, 6911, 6917, \
             6947, 6949, 6959, 6961, 6967, 6971, 6977, 6983, 6991, 6997, \
             7001, 7013, 7019, 7027, 7039, 7043, 7057, 7069, 7079, 7103, \
             7109, 7121, 7127, 7129, 7151, 7159, 7177, 7187, 7193, 7207, \
             7211, 7213, 7219, 7229, 7237, 7243, 7247, 7253, 7283, 7297, \
             7307, 7309, 7321, 7331, 7333, 7349, 7351, 7369, 7393, 7411, \
             7417, 7433, 7451, 7457, 7459, 7477, 7481, 7487, 7489, 7499, \
             7507, 7517, 7523, 7529, 7537, 7541, 7547, 7549, 7559, 7561, \
             7573, 7577, 7583, 7589, 7591, 7603, 7607, 7621, 7639, 7643, \
             7649, 7669, 7673, 7681, 7687, 7691, 7699, 7703, 7717, 7723, \
             7727, 7741, 7753, 7757, 7759, 7789, 7793, 7817, 7823, 7829, \
             7841, 7853, 7867, 7873, 7877, 7879, 7883, 7901, 7907, 7919, \
             7927, 7933, 7937, 7949, 7951, 7963, 7993, 8009, 8011, 8017, \
             8039, 8053, 8059, 8069, 8081, 8087, 8089, 8093, 8101, 8111, \
             8117, 8123, 8147, 8161, 8167, 8171, 8179, 8191, 8209, 8219, \
             8221, 8231, 8233, 8237, 8243, 8263, 8269, 8273, 8287, 8291, \
             8293, 8297, 8311, 8317, 8329, 8353, 8363, 8369, 8377, 8387, \
             8389, 8419, 8423, 8429, 8431, 8443, 8447, 8461, 8467, 8501, \
             8513, 8521, 8527, 8537, 8539, 8543, 8563, 8573, 8581, 8597, \
             8599, 8609, 8623, 8627, 8629, 8641, 8647, 8663, 8669, 8677, \
             8681, 8689, 8693, 8699, 8707, 8713, 8719, 8731, 8737, 8741, \
             8747, 8753, 8761, 8779, 8783, 8803, 8807, 8819, 8821, 8831, \
             8837, 8839, 8849, 8861, 8863, 8867, 8887, 8893, 8923, 8929, \
             8933, 8941, 8951, 8963, 8969, 8971, 8999, 9001, 9007, 9011, \
             9013, 9029, 9041, 9043, 9049, 9059, 9067, 9091, 9103, 9109, \
             9127, 9133, 9137, 9151, 9157, 9161, 9173, 9181, 9187, 9199, \
             9203, 9209, 9221, 9227, 9239, 9241, 9257, 9277, 9281, 9283, \
             9293, 9311, 9319, 9323, 9337, 9341, 9343, 9349, 9371, 9377, \
             9391, 9397, 9403, 9413, 9419, 9421, 9431, 9433, 9437, 9439, \
             9461, 9463, 9467, 9473, 9479, 9491, 9497, 9511, 9521, 9533, \
             9539, 9547, 9551, 9587, 9601, 9613, 9619, 9623, 9629, 9631, \
             9643, 9649, 9661, 9677, 9679, 9689, 9697, 9719, 9721, 9733, \
             9739, 9743, 9749, 9767, 9769, 9781, 9787, 9791, 9803, 9811, \
             9817, 9829, 9833, 9839, 9851, 9857, 9859, 9871, 9883, 9887, \
             9901, 9907, 9923, 9929, 9931, 9941, 9949, 9967, 9973,10007, \
            10009,10037,10039,10061,10067,10069,10079,10091,10093,10099, \
            10103,10111,10133,10139,10141,10151,10159,10163,10169,10177, \
            10181,10193,10211,10223,10243,10247,10253,10259,10267,10271, \
            10273,10289,10301,10303,10313,10321,10331,10333,10337,10343, \
            10357,10369,10391,10399,10427,10429,10433,10453,10457,10459, \
            10463,10477,10487,10499,10501,10513,10529,10531,10559,10567, \
            10589,10597,10601,10607,10613,10627,10631,10639,10651,10657, \
            10663,10667,10687,10691,10709,10711,10723,10729,10733,10739, \
            10753,10771,10781,10789,10799,10831,10837,10847,10853,10859, \
            10861,10867,10883,10889,10891,10903,10909,10937,10939,10949, \
            10957,10973,10979,10987,10993,11003,11027,11047,11057,11059, \
            11069,11071,11083,11087,11093,11113,11117,11119,11131,11149, \
            11159,11161,11171,11173,11177,11197,11213,11239,11243,11251, \
            11257,11261,11273,11279,11287,11299,11311,11317,11321,11329, \
            11351,11353,11369,11383,11393,11399,11411,11423,11437,11443, \
            11447,11467,11471,11483,11489,11491,11497,11503,11519,11527, \
            11549,11551,11579,11587,11593,11597,11617,11621,11633,11657, \
            11677,11681,11689,11699,11701,11717,11719,11731,11743,11777, \
            11779,11783,11789,11801,11807,11813,11821,11827,11831,11833, \
            11839,11863,11867,11887,11897,11903,11909,11923,11927,11933, \
            11939,11941,11953,11959,11969,11971,11981,11987,12007,12011, \
            12037,12041,12043,12049,12071,12073,12097,12101,12107,12109, \
            12113,12119,12143,12149,12157,12161,12163,12197,12203,12211, \
            12227,12239,12241,12251,12253,12263,12269,12277,12281,12289, \
            12301,12323,12329,12343,12347,12373,12377,12379,12391,12401, \
            12409,12413,12421,12433,12437,12451,12457,12473,12479,12487, \
            12491,12497,12503,12511,12517,12527,12539,12541,12547,12553, \
            12569,12577,12583,12589,12601,12611,12613,12619,12637,12641, \
            12647,12653,12659,12671,12689,12697,12703,12713,12721,12739, \
            12743,12757,12763,12781,12791,12799,12809,12821,12823,12829, \
            12841,12853,12889,12893,12899,12907,12911,12917,12919,12923, \
            12941,12953,12959,12967,12973,12979,12983,13001,13003,13007, \
            13009,13033,13037,13043,13049,13063,13093,13099,13103,13109, \
            13121,13127,13147,13151,13159,13163,13171,13177,13183,13187, \
            13217,13219,13229,13241,13249,13259,13267,13291,13297,13309, \
            13313,13327,13331,13337,13339,13367,13381,13397,13399,13411, \
            13417,13421,13441,13451,13457,13463,13469,13477,13487,13499 ] )
        
          if ( n < 0 or prime_max <= n ):
            print ( '' )
            print ( 'PRIME - Fatal error!' )
            print ( '  0 <= N < %d' % ( prime_max ) )
            exit ( 'PRIME - Fatal error!' )
        
          return prime_vector[n]      
          
        hamm_seq = hammersley_sequence(i1, i2, m, n)
        samp = dist.ppf(hamm_seq)
        
        if vectorized == False:
            return(samp)
        
        else:  
            samp = samp.reshape(1, -1)
            return(samp)
         
def Rosenblatt_Transform(dist, x_i):
    """
    Computes the equivalent normal distribution for a non-normal random variates
    given the variates value.

    Parameters
    ----------
    dist: Object distribution made with makedist function.
    
    x_i: Random variates value.

    Returns
    -------
    x_N_mean: Equivalent normal mean
    
    x_N_std: Equivalent normal standard deviation

    """   
    if dist.stats(moments = 's') > 1 or dist.stats(moments = 's') < -1:
        
        x_N_mean = dist.median()
        x_N_std  = (x_i - x_N_mean)/sst.norm.ppf(dist.cdf(x_i))
        
        return(x_N_mean, x_N_std)
    
    else:
        x_N_std  = sst.norm.pdf(sst.norm.ppf(dist.cdf(x_i)))/dist.pdf(x_i)
        x_N_mean = x_i - sst.norm.ppf(dist.cdf(x_i))*x_N_std
        return(x_N_mean, x_N_std)
    
def form(func, dist_list, init_search_point, alg):
    """
    First-order reliability analysis function.

    Parameters
    ----------
    func: Limit state function.
    
    dist_list: List with the distributions of each variable in the limit state function.
    
    init_search_point: Initial search point for the algorithm (this can affect the solution extremely in a matter of convergence.) 
    
    alg: String with the name of the optimization algorithm to use, it may be:
        
        'slsqp': Sequential Least Square Programming from scipy;
        
        'HL-R': Hasofer-Lind-Rackwitz algorithm;
        
        'HL-RF': Hasofer-Lind-Rackwitz-Fiessler algorithm (Newton-Raphson type method).

    Returns
    -------
    beta_value: Reliability index;
    
    p_f: Failure probability;
    
    x: Design point in the original space;
    
    u: Design point in the standard normal space;
    
    mu: Global normal mean array;
    
    sig: Global standard deviation array;
    
    cosines: Direction cosines (important for  the Second-Order Reliability method)
    """
    
    def SLSQP(func, dist_list, init_search_point):
 
        dim = len(dist_list)
        current_beta = 0
        new_beta = 1
        sig = np.empty((1, dim))
        mu  = np.empty((1, dim))
        new_search_point = np.array(init_search_point).reshape((1, dim))
        
        def f_l(x_l):
            return(func([x_l[i,:]*sig[0,i] + mu[0,i] for i in range(0, dim)]))
        
        while abs(current_beta-new_beta) > 0.001:
            current_search_point = new_search_point
            current_beta = new_beta
            for i in range(0, dim):
                if dist_list[i][1] != 'norm':
                    mu[0,i], sig[0, i] = Rosenblatt_Transform(dist_list[i][0], current_search_point[0,i])
                else:
                    mu[0,i], sig[0, i] = dist_list[i][0].mean(), dist_list[i][0].std()
            
            dist_fun = lambda u: np.linalg.norm(u) 
        
            alg = 'SLSQP'
            
            H = lambda u: f_l(u)
            cons = ({'type': 'ineq', 'fun': lambda u: -(H(u.reshape(-1,1)))})
            
            result = scipy.optimize.minimize(dist_fun, x0 = current_search_point, constraints = cons, method=alg)
            
            new_beta = result.fun
            u = np.array(result.x).reshape((1,dim))
            
            new_search_point = np.empty((1, dim))
            for i in range(0, dim):
                new_search_point[0,i] = mu[0,i] + u[0,i]*sig[0,i]
                
        beta_value = new_beta
            
        p_f = sst.norm.cdf(-beta_value)
        iterations = result.nit
        u = result.x
        x = u[:]*sig[0,:] + mu[0,:]
        print(u)
        grad_val = scipy.optimize.approx_fprime(x, func, 0.00000001)
        grad_val = grad_val.reshape((1, dim))
         
        sum1 = np.sum((grad_val[0,:]**2)*(sig[0,:]**2))
        cosines = np.empty((1, dim))
         
        for i in range(0, dim):
            cosines[0,i] = grad_val[0,i]*sig[0,i]/np.sqrt(sum1)       
        
        print('------------------------')
        print('First-Order Reliability Analysis')
        print('Algorithm: slsqp solver')
        print('Iterations: {}\nReliability index = {}\nProbability of failure = {}'.format(iterations, beta_value, p_f))
        print('------------------------')
    
        return(beta_value, p_f, x, u, mu, sig, cosines)  
    
    def HL_R(func, dist_list, init_search_point):
 
        iterations = 0
        cur_beta = 3
        new_beta = 0
        dim = len(dist_list)
        global_mean_arr = np.empty((1, dim))
        global_std_arr  = np.empty((1, dim))
        new_search_point = np.array(init_search_point).reshape((1, dim))
         
        while abs(cur_beta - new_beta) > 0.001:
            cur_beta = new_beta
            cur_cosines = np.zeros((1, dim))
            new_cosines = np.ones((1, dim))
             
            while max((abs(cur_cosines - new_cosines))[0]) > 0.005:
                 
                cur_cosines = new_cosines
                
                cur_search_point = new_search_point
                 
                for i in range(0, dim):
                    if dist_list[i][1] != 'norm':
                        global_mean_arr[0, i], global_std_arr[0, i] = Rosenblatt_Transform(dist_list[i][0], cur_search_point[0,i])
                    else:
                        global_mean_arr[0, i], global_std_arr[0, i] = dist_list[i][0].mean(), dist_list[i][0].std()

                grad_val = scipy.optimize.approx_fprime(cur_search_point[0], func, 0.00000001)
                grad_val = grad_val.reshape((1, dim))
                 
                sum1 = np.sum((grad_val[0,:]**2)*(global_std_arr[0,:]**2))
                cosines = np.empty((1, dim))
                 
                for i in range(0, dim):
                    cosines[0,i] = grad_val[0,i]*global_std_arr[0,i]/np.sqrt(sum1)
                 
                new_cosines = cosines
                new_search_point = np.empty((1, dim))
                for i in range(0, dim):
                    new_search_point[0,i] = global_mean_arr[0,i] - new_cosines[0,i]*global_std_arr[0,i]*cur_beta
                
                iterations = iterations + 1
                 
               
            B = Symbol('B')
            coordinates = []
            for i in range(0, dim):
                coordinates.append(global_mean_arr[0, i] - new_cosines[0,i]*global_std_arr[0, i]*B)
            
            print(coordinates)
            new_beta = float(solve(func(coordinates), B)[0])
        
        x = new_search_point
        
        def f_l(x_l):
            return(func([x_l[i]*global_std_arr[0,i] + global_mean_arr[0,i] for i in range(0, dim)]))
        
        if f_l(np.zeros(dim)) > 0:
            beta_value = -new_beta
        else:
            beta_value = new_beta
            
        cosines = new_cosines    
        p_f = sst.norm.cdf(-beta_value)
        
        print(x)
        u = (x[0,:] - global_mean_arr[0,:])/global_std_arr
        
        print('-------------------------')
        print('First-Order Reliability Analysis')
        print('Algorithm: HL-R solver')
        print('Iterations: {}\nReliability index = {}\nProbability of failure = {}'.format(iterations, beta_value, p_f))
        print('-------------------------')
        
        return(beta_value, p_f, x, u, global_mean_arr, global_std_arr, cosines)
    
    def HL_RF(func, dist_list, init_search_point):

        cur_beta = 3
        new_beta = 0
        dim = len(dist_list)

        new_search_point = np.array(init_search_point).reshape((1, dim))
        
        
        iterations = 0
        while abs(cur_beta - new_beta) > 0.00001 and abs(func(new_search_point[0])) > 0.00001:
            global_mean_arr = np.empty((1, dim))
            global_std_arr  = np.empty((1, dim))
            cur_beta = new_beta
            cur_search_point = new_search_point
            
            for i in range(0, dim):
                if dist_list[i][1] != 'norm':
                    global_mean_arr[0,i], global_std_arr[0, i] = Rosenblatt_Transform(dist_list[i][0], cur_search_point[0,i])
                else:
                    global_mean_arr[0,i], global_std_arr[0, i] = dist_list[i][0].mean(), dist_list[i][0].std()
            
            
            f_val = func(cur_search_point[0])
            
            x_ast = np.empty((1, dim))
            for i in range(0, dim):
                x_ast[0,i] =(cur_search_point[0,i] - global_mean_arr[0,i])/global_std_arr[0,i]

            grad_val = scipy.optimize.approx_fprime(cur_search_point[0], func, 0.000001)
            grad_val = grad_val.reshape((1, dim))           
            
            grad_val_ast = np.empty(grad_val.shape)
            for i in range(0, dim):
                grad_val_ast[0,i] = grad_val[0,i]*global_std_arr[0,i]
            
            t1 = 1/np.sum(grad_val_ast[0,:]**2)

            t2 = sum(grad_val_ast[0,:]*x_ast[0,:]) - f_val
            
            t3 = t1*t2
            
            new_x_ast = np.empty(x_ast.shape)
            for i in range(0, dim):
                new_x_ast[0,i] = t3*grad_val_ast[0,i]
            u = new_x_ast
            new_beta = np.linalg.norm(new_x_ast)
           
            new_search_point = np.empty((1, dim))
            for i in range(0, dim):
                new_search_point[0,i] = new_x_ast[0,i]*global_std_arr[0,i] + global_mean_arr[0,i]
            iterations = iterations + 1
        
        grad_val_ast_sum = sum(grad_val_ast[0,:]**2)
        cosines = grad_val_ast/(grad_val_ast_sum**0.5)
        def f_l(x_l):
            return(func([x_l[i]*global_std_arr[0,i] + global_mean_arr[0,i] for i in range(0, dim)]))
        
        if f_l(np.zeros(dim)) > 0:
            beta_value = -new_beta
        else:
            beta_value = new_beta
        x = new_search_point
        print(x)
        p_f = sst.norm.cdf(-beta_value)
        
        print('-------------------------')
        print('First-Order Reliability Analysis')
        print('Algorithm: HL-RF solver')
        print('Iterations: {}\nReliability index = {}\nProbability of failure = {}'.format(iterations, beta_value, p_f))
        print('-------------------------')
        
        return(beta_value, p_f, x, u, global_mean_arr, global_std_arr, cosines)
    
    if alg == 'slsqp':
        return(SLSQP(func, dist_list, init_search_point))
    elif alg == 'HL-R':
        return(HL_R(func, dist_list, init_search_point))
    elif alg == 'HL-RF':
        return(HL_RF(func, dist_list, init_search_point))
    
def sorm(func, dist_list, init_search_point, alg):
    """
    Second-order reliability analysis function.

    Parameters
    ----------
    func: Limit state function.
    
    dist_list: List with the distributions of each variable in the limit state function.
    
    init_search_point: Initial search point for the algorithm (this can affect the solution extremely in a matter of convergence.) 
    
    alg: String with the name of the optimization algorithm to use, it may be:
        
        'slsqp': Sequential Least Square Programming from scipy;
        
        'HL-R': Hasofer-Lind-Rackwitz algorithm;
        
        'HL-RF': Hasofer-Lind-Rackwitz-Fiessler algorithm (Newton-Raphson type method).

    Returns
    -------
    beta_sorm: Reliability index for the second-order reliability method;
    
    p_f_sorm: Failure probability.
    """    
    def SLSQP(func, dist_list, init_search_point):
 
        dim = len(dist_list)
        current_beta = 0
        new_beta = 1
        sig = np.empty((1, dim))
        mu  = np.empty((1, dim))
        new_search_point = np.array(init_search_point).reshape((1, dim))
        
        def f_l(x_l):
            return(func([x_l[i,:]*sig[0,i] + mu[0,i] for i in range(0, dim)]))
        
        while abs(current_beta-new_beta) > 0.001:
            current_search_point = new_search_point
            current_beta = new_beta
            for i in range(0, dim):
                if dist_list[i][1] != 'norm':
                    mu[0,i], sig[0, i] = Rosenblatt_Transform(dist_list[i][0], current_search_point[0,i])
                else:
                    mu[0,i], sig[0, i] = dist_list[i][0].mean(), dist_list[i][0].std()
            
            dist_fun = lambda u: np.linalg.norm(u) 
        
            alg = 'SLSQP'
            
            H = lambda u: f_l(u)
            cons = ({'type': 'eq', 'fun': lambda u: -(H(u.reshape(-1,1)))})
            
            result = scipy.optimize.minimize(dist_fun, x0 = current_search_point, constraints = cons, method=alg)
            
            new_beta = result.fun
            u = np.array(result.x).reshape((1,dim))
            
            new_search_point = np.empty((1, dim))
            for i in range(0, dim):
                new_search_point[0,i] = mu[0,i] + u[0,i]*sig[0,i]
                
        beta_value = new_beta    
        p_f = sst.norm.cdf(-beta_value)
        iterations = result.nit
        u = result.x
        x = u[:]*sig[0,:] + mu[0,:]
        grad_val = scipy.optimize.approx_fprime(x, func, 0.00000001)
        grad_val = grad_val.reshape((1, dim))
         
        sum1 = np.sum((grad_val[0,:]**2)*(sig[0,:]**2))
        cosines = np.empty((1, dim))
         
        for i in range(0, dim):
            cosines[0,i] = grad_val[0,i]*sig[0,i]/np.sqrt(sum1)       
    
        return(beta_value, p_f, x, u, mu, sig, cosines, iterations)  
    
    def HL_R(func, dist_list, init_search_point):
 
        iterations = 0
        cur_beta = 3
        new_beta = 0
        dim = len(dist_list)
        global_mean_arr = np.empty((1, dim))
        global_std_arr  = np.empty((1, dim))
        new_search_point = np.array(init_search_point).reshape((1, dim))
         
        while abs(cur_beta - new_beta) > 0.001:
            cur_beta = new_beta
            cur_cosines = np.zeros((1, dim))
            new_cosines = np.ones((1, dim))
             
            while max((abs(cur_cosines - new_cosines))[0]) > 0.005:
                 
                cur_cosines = new_cosines
                
                cur_search_point = new_search_point
                 
                for i in range(0, dim):
                    if dist_list[i][1] != 'norm':
                        global_mean_arr[0, i], global_std_arr[0, i] = Rosenblatt_Transform(dist_list[i][0], cur_search_point[0,i])
                    else:
                        global_mean_arr[0, i], global_std_arr[0, i] = dist_list[i][0].mean(), dist_list[i][0].std()
                  
                
                grad_val = scipy.optimize.approx_fprime(cur_search_point[0], func, 0.00000001)
                grad_val = grad_val.reshape((1, dim))
                 
                sum1 = np.sum((grad_val[0,:]**2)*(global_std_arr[0,:]**2))
                cosines = np.empty((1, dim))
                 
                for i in range(0, dim):
                    cosines[0,i] = grad_val[0,i]*global_std_arr[0,i]/np.sqrt(sum1)
                 
                new_cosines = cosines
                new_search_point = np.empty((1, dim))
                for i in range(0, dim):
                    new_search_point[0,i] = global_mean_arr[0,i] - new_cosines[0,i]*global_std_arr[0,i]*cur_beta
                
                iterations = iterations + 1
                 
               
            B = Symbol('B')
            coordinates = []
            for i in range(0, dim):
                coordinates.append(global_mean_arr[0, i] - new_cosines[0,i]*global_std_arr[0, i]*B)
            new_beta = float(solve(func(coordinates), B)[0])
            
        cosines = new_cosines    
        beta_value = new_beta
        p_f = sst.norm.cdf(-new_beta)
        x = new_search_point
        u = (x[0,:] - global_mean_arr[0,:])/global_std_arr
    
        return(beta_value, p_f, x, u, global_mean_arr, global_std_arr, cosines, iterations)
    
    def HL_RF(func, dist_list, init_search_point):

        cur_beta = 3
        new_beta = 0
        dim = len(dist_list)

        new_search_point = np.array(init_search_point).reshape((1, dim))
        iterations = 0
        while abs(cur_beta - new_beta) > 0.001 and abs(func(new_search_point[0])) > 0.001:
            global_mean_arr = np.empty((1, dim))
            global_std_arr  = np.empty((1, dim))
            cur_beta = new_beta
            cur_search_point = new_search_point
            
            for i in range(0, dim):
                if dist_list[i][1] != 'norm':
                    global_mean_arr[0,i], global_std_arr[0, i] = Rosenblatt_Transform(dist_list[i][0], cur_search_point[0,i])
                else:
                    global_mean_arr[0,i], global_std_arr[0, i] = dist_list[i][0].mean(), dist_list[i][0].std()
           
            f_val = func(cur_search_point[0])
            
            x_ast = np.empty((1, dim))
            for i in range(0, dim):
                x_ast[0,i] =(cur_search_point[0,i] - global_mean_arr[0,i])/global_std_arr[0,i]

            grad_val = scipy.optimize.approx_fprime(cur_search_point[0], func, 0.000001)
            grad_val = grad_val.reshape((1, dim))           
            
            grad_val_ast = np.empty(grad_val.shape)
            for i in range(0, dim):
                grad_val_ast[0,i] = grad_val[0,i]*global_std_arr[0,i]
            
            t1 = 1/np.sum(grad_val_ast[0,:]**2)

            t2 = sum(grad_val_ast[0,:]*x_ast[0,:]) - f_val
            
            t3 = t1*t2
            
            new_x_ast = np.empty(x_ast.shape)
            for i in range(0, dim):
                new_x_ast[0,i] = t3*grad_val_ast[0,i]
            u = new_x_ast
            new_beta = np.linalg.norm(new_x_ast)
           
            new_search_point = np.empty((1, dim))
            for i in range(0, dim):
                new_search_point[0,i] = new_x_ast[0,i]*global_std_arr[0,i] + global_mean_arr[0,i]
            iterations = iterations + 1
        
        grad_val_ast_sum = sum(grad_val_ast[0,:]**2)
        cosines = grad_val_ast/(grad_val_ast_sum**0.5)
        beta_value = new_beta
        x = new_search_point
        p_f = sst.norm.cdf(-beta_value)
        
        return(beta_value, p_f, x, u, global_mean_arr, global_std_arr, cosines, iterations)
    
    if alg == 'slsqp':
        beta_value, p_f, x, u, mu, sig, cosines, iterations = SLSQP(func, dist_list, init_search_point)
    elif alg == 'HL-R':
        beta_value, p_f, x, u, mu, sig, cosines, iterations = HL_R(func, dist_list, init_search_point)
    elif alg == 'HL-RF':
        beta_value, p_f, x, u, mu, sig, cosines, iterations = HL_RF(func, dist_list, init_search_point)
    
    d = len(dist_list)

    R0 = np.eye(d)
    
    for i in range(0, d):
        R0[-1,i] = cosines[0,i]
     
    Q, R = scipy.linalg.rq(R0)
    
    def f_l(x_l):
        return(func([x_l[i]*sig[0,i] + mu[0,i] for i in range(0, d)]))
    
    x = np.array(x).reshape((1, -1))
    u = x[0,:]*sig[0,:] + mu[0,:]
    
    H = nd.Hessian(f_l)(u)
    
    grad_val_standard = (scipy.optimize.approx_fprime(x[0], func, 0.00000001)[:])*(sig[0,:])
    
    dist_standard = np.linalg.norm(grad_val_standard)
    
    A_1 = 1/dist_standard
    R_transp = np.transpose(R)
    A_2 = R.dot(H)
    A_3 = A_2.dot(R_transp)
    
    A = A_3.dot(A_1)
    
    A = A[0:-1, 0:-1]
    
    k = np.linalg.eig(A)[0]
    
    prod_arr = np.empty((1, len(k)))
    for i in range(0, len(k)):
        prod_arr[0,i] = (1 + beta_value*k[i])**-0.5
        
    p_f_sorm = p_f*np.prod(prod_arr)
    beta_sorm = -1*scipy.stats.norm.ppf(p_f_sorm)
    
    print('-------------------------')
    print('Second-Order Reliability Analysis')
    print('Algorithm:',alg,'solver')
    print('Iterations: {}\nReliability index = {}\nProbability of failure = {}'.format(iterations, beta_sorm, p_f_sorm))
    print('-------------------------')
    
    return(beta_sorm, p_f_sorm)

def pfsim(f, dist_list, N_cycles, method):
    
    """
    Compute the failure probability by sampling.

    Parameters
    ----------
    f: Limit state function;
    
    dist_list: List with the distributions of each variable in the limit state function.;
    
    N_cycles: Number of cycles for simulating the samples;
     
    method: Sampling method: 'mc' for Monte-Carlo and 'lh' for Latin Hypercube.

    Returns
    -------
    p_f: Failure probability.

    """
    dim = len(dist_list)
    samp_sim = np.empty((dim, N_cycles))
    
    if method == 'mc':
        for i in range(0, dim):
            samp_sim[i,:] = mc_design(dist_list[i]).gen_samp(N_cycles)   
                
    elif method == 'lh':
        for i in range(0, dim):
            samp_sim[i,:] = lh_design(dist_list[i]).gen_samp(1, N_cycles)
        
    count = 0
    for N in range(0, N_cycles):
        f_eval = f(samp_sim[:, N])
        if f_eval > 0:
            count = count + 1
             
    p_f = count/N_cycles
    
    return(p_f)
            

def mcsens(func, dist_list, N_cycles, estimator = 'owen'):
    d = len(dist_list)
    if estimator == 'owen':
        Z = np.empty((N_cycles, d))
        for i in range(0, d):
            Z[:,i] = mc_design(dist_list[i]).gen_samp(N_cycles)
            
        Z_l = np.empty((N_cycles, d, d))
        for i in range(0, d):
            for j in range(0, d):
                Z_l[:,j,i] = mc_design(dist_list[j]).gen_samp(N_cycles)
                
        
        fZ = np.empty((1,N_cycles))
        for i in range(0, N_cycles):
            fZ[0,i] = func(Z[i,:])
            
        E_fZ = np.mean(fZ[0,:])
        
        Var_fZ = np.var(fZ[0,:], ddof = 1)
        
        V_j = np.empty((1, d))
        for j in range(0, d):
            fZ_li = np.empty((1, N_cycles))
            
            for i in range(0, N_cycles):
                fZ_li[0,i] = func(Z_l[i,:,j])  
                
            E_fZ_l = np.mean(fZ_li[0,:])  
            Var_fZ_l = np.var(fZ_li[0,:], ddof = 1)
            
            V_j[0,j] = (2*N_cycles/(2*N_cycles - 1))*(N_cycles**-1 * sum(fZ[0,:]*fZ_li[0,:]) - ((E_fZ + E_fZ_l)/2)**2 + ((Var_fZ + Var_fZ_l)/(4*N_cycles)))
       
        s_j = V_j[0,:]/Var_fZ
        
        return(s_j)
            
              
        
    