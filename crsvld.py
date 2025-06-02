import os 
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import random
from random import randrange
import datetime
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb


from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from itertools import product

import warnings
warnings.filterwarnings('ignore')

from llf_core import overwrite_tree_response, predict_single_sample, \
build_regression_tree,uniqueness_not_ok,find_best_split,mean_loss,solve_llf_at_x0,weighted_ridge_regression

from llf_testing import generate_random_table,paper_data,generate_random_lin_table,\
paper_data2,paper_data2_gamma,paper_data2_poisson

from llf_testing import make_llf_model,batch_solve_llf,compare_models
from llf_testing import LLTree,LLForest

timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') 
M, N = 1000, 20  # Adjust as needed
run_signature = timestamp + '_with_' + str(N) + '_cols_' + str(M)+ '_rows'
metadata = {'train_proportion':0.5}

df = generate_random_table(M, N, response_type='continuous')
Y_vec = df[['Response']]
X_vec = df.drop(columns = ['Response'])



p_df = paper_data(M, N, response_type='continuous')
p_Y_vec = p_df[['Response']]
p_X_vec = p_df.drop(columns = ['Response'])

p_df_hvar = paper_data(M, N, response_type='continuous',low_var=False)
phvar_Y_vec = p_df_hvar[['Response']]
phvar_X_vec = p_df_hvar.drop(columns = ['Response'])



p2g_df = paper_data2_gamma(M, N)
p2g_Y_vec = p2g_df[['Response']]
p2g_X_vec = p2g_df.drop(columns=['Response'])

p2g_df_hvar = paper_data2_gamma(M, N,low_var=False)
p2ghvar_Y_vec = p2g_df_hvar[['Response']]
p2ghvar_X_vec = p2g_df_hvar.drop(columns=['Response'])



p2ps_df = paper_data2_poisson(M, N)
p2ps_Y_vec = p2ps_df[['Response']]
p2ps_X_vec = p2ps_df.drop(columns=['Response'])



p2_df = paper_data2(M, N, response_type='continuous')
p2_Y_vec = p2_df[['Response']]
p2_X_vec = p2_df.drop(columns=['Response'])

p2hvar_df = paper_data2(M, N, response_type='continuous',low_var=False)
p2hvar_Y_vec = p2hvar_df[['Response']]
p2hvar_X_vec = p2hvar_df.drop(columns=['Response'])

lin_df = generate_random_lin_table(M, N, response_type='continuous')
lin_Y_vec = lin_df[['Response']]
lin_X_vec = lin_df.drop(columns=['Response'])


max_depth_grid = [4]
forest_lambda_grid = [0.1, 0.5,2,4]
ridge_lambda_grid = [0.1, 1, 5]
min_leaf_size_grid = [10, 20]
omega_grid = [0.05, 0.2]
n_estimators_grid = [10,20]


param_grid = list(product(forest_lambda_grid,n_estimators_grid,ridge_lambda_grid,max_depth_grid,min_leaf_size_grid, omega_grid))
X = p2ps_X_vec
Y = p2ps_Y_vec
dataset = 'pois pub2, lin splits'
print(dataset)
actual_param_grid = []
with open('run_experiments.json') as wf:
    experiments = json.load(wf)
    if dataset not in experiments.keys():
        experiments[dataset] = {}
    for param_combo in param_grid:
        if str(param_combo) not in experiments[dataset].keys():
            actual_param_grid.append(param_combo)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

random.shuffle(actual_param_grid)

for params in actual_param_grid:
    warnings.filterwarnings('ignore')
    print(params)
    forest_lambda,n_estimators,ridge_lambda,max_depth, min_leaf_size, omega = params
    print("nr est:",n_estimators)
    str_params = str(params)
    fold_scores = []
    
    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        Y_train, Y_val = Y.iloc[train_index], Y.iloc[val_index]
        
        
        # REMEMBER TO FUCKING CHANGE THE SPLITTING MODE YOU ABSOLUTE DONKEY
        forest = LLForest(X_train, Y_train,ridge_lambda=ridge_lambda, max_depth=max_depth, \
                          min_leaf_size=min_leaf_size,n_estimators=n_estimators,omega=omega,mode='lin')
        forest.train()
        
        preds = []
        length = X_val.shape[0]
        for row in range(0,length):
            if row == np.floor(length/2):
                print(row)
            x0 = X_val.iloc[row]
            
            forest_weights = forest.calculate_forest_weights(x0)
            mutest, thetatest = solve_llf_at_x0(x0,X_train,Y_train,alphas_vec=forest_weights,forest_lambda=forest_lambda)
            try:
                preds.append(mutest[0])
            except:
                preds.append(mutest)
        
        score = np.sqrt(mean_squared_error(Y_val, preds))  # or other metric
        fold_scores.append(score)
        
    avg_score = np.mean(fold_scores)
    results.append((params, avg_score))
    experiments[dataset][str_params] = {'fold_scores':fold_scores,'avg_score':avg_score}
    with open('run_experiments.json','w') as wf:
        json.dump(experiments,wf,indent=4)
    wf.close()


# Find best params, lowest error
best_params, best_score = min(results, key=lambda x: x[1])
print("Best parameters:", best_params)
print("Best average CV score:", best_score)