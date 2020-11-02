# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 20:32:22 2020

@author: iago
"""
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
import matplotlib.pyplot as plt
import numpy as np
import openturns as ot
from scipy.stats import kstest, genextreme
import threshmodels2 as tm2

raw_data = np.array(np.loadtxt(fname='IRIS_data.txt')) #loading dataset

nom_thick = 2.10 #nominal thickness
min_thick = 0.1
years = 4

n_tubes = len(raw_data) #number of tubes/population

loss_data = sorted(nom_thick - raw_data) #wall loss dataset

#----Creating POT model for modeling excesses------#
pot_model = tm2.POT(loss_data) 
pot_model.mrl()
pot_model.parst()
u = 0.5
#--------------------------------------------------#

#----Building GPD model-----#
gpd_model = tm2.build_model(loss_data, u)
gpd_model.disp()
#---------------------------#

#-----Model Diagnosis-----#
diag = tm2.model_diag(gpd_model)
diag.pdf()
diag.cdf()
diag.qqplot()
diag.ppplot()
#-------------------------#

#----Model extrapolation----#
retlev = tm2.retlev_tool(gpd_model, 0.05)
retlev.retlev_plot(1, 500)
#---------------------------#

#----Failure probability computation----#
cr_arr = [ldata/years for ldata in loss_data]

X_nom = [nom_thick/cr for cr in cr_arr]
fit_nom = genextreme.fit(X_nom)

X_min = [min_thick/cr for cr in cr_arr]
fit_min = genextreme.fit(X_min)

ks_nom = kstest(X_nom, 'genextreme', fit_nom)
ks_min = kstest(X_min, 'genextreme', fit_min)


X_nom_dist = ot.GeneralizedExtremeValue(fit_nom[1], fit_nom[2], -fit_nom[0])
X_min_dist = ot.GeneralizedExtremeValue(fit_min[1], fit_min[2], -fit_min[0])

marginals = [X_nom_dist, X_min_dist]

RS = ot.CorrelationMatrix(2)
# Evaluate the correlation matrix of the Normal copula from RS
R = ot.NormalCopula.GetCorrelationFromSpearmanCorrelation(RS)
# Create the Normal copula parametrized by RS
copula = ot.NormalCopula(R)
distribution = ot.ComposedDistribution(marginals, copula)
distribution.setDescription(['X_nom', 'X_min'])

# create the model
model = ot.SymbolicFunction(['X_nom', 'X_min'], ['X_nom - X_min'])

vect = ot.RandomVector(distribution)
G = ot.CompositeRandomVector(model, vect)
event = ot.ThresholdEvent(G, ot.Less(), 0)
event.setName("deviation")


Define a solver
optimAlgo = ot.Cobyla()
optimAlgo.setMaximumEvaluationNumber(1000)
optimAlgo.setMaximumAbsoluteError(1.0e-10)
optimAlgo.setMaximumRelativeError(1.0e-10)
optimAlgo.setMaximumResidualError(1.0e-10)
optimAlgo.setMaximumConstraintError(1.0e-10)

algo = ot.FORM(optimAlgo, event, distribution.getMean())
algo.run()
result = algo.getResult()

p_f = result.getEventProbability()

uqt_nom = uqt.makedist('genextreme', loc = 11.6975, scale = 3.95809 , c = 0.0869234)
uqt_min = uqt.makedist('genextreme', loc = 0.557022, scale = 0.188481, c = 0.0869234)


dist_list = [uqt_nom, uqt_min]
init_search_point = [np.mean(X_nom), np.mean(X_min)]

def func(X):
    return(X[0] - X[1])

plt.figure(1)
x_2 = np.arange(-10,10,0.1)
x_2_nom = []
x_2_min = []
for i in range(0, len(x_2)):
    nom_mean, nom_std = uqt.Rosenblatt_Transform(uqt_nom[0], x_2[i])
    x_2_nom.append((x_2[i] - nom_mean)/nom_std)
    min_mean, min_std = uqt.Rosenblatt_Transform(uqt_min[0], x_2[i])
    x_2_min.append((x_2[i] - min_mean)/min_std)

plt.plot(x_2_nom, x_2_min)
plt.plot([0, u[0,0]],[0,u[0,1]], label = 'HLR Result', color = 'black')
plt.scatter(0,0, label = 'U-Space origin', c = 'gray')
plt.scatter(u[0,0], u[0,1], label = 'Design point', c = 'black')
plt.legend()
plt.xlabel(r'$U_{nom}$')
plt.ylabel(r'$U_{min}$')

#-------------------------------------

#-------Prognosis analysis--------#
current_pf = X_nom_dist.computeCDF(4)
year_of_failure = X_nom_dist.computeQuantile(p_f)
RUL = year_of_failure[0] - years
print(RUL)
#---------------------------------#
