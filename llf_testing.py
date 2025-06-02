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
import inspect
from numba import jit

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import statsmodels.api as sm


from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


from llf_core import overwrite_tree_response, predict_single_sample, \
build_regression_tree,uniqueness_not_ok,find_best_split,mean_loss,solve_llf_at_x0,weighted_ridge_regression

def generate_random_table(M: int, N: int, response_type='continuous'):
    """
    Generates a random dataset with M rows and N independent variables plus one response variable.
    
    Parameters:
    M (int): Number of rows (observations)
    N (int): Number of independent variables
    response_type (str): Type of response variable ('continuous' or 'categorical')
    
    Returns:
    pd.DataFrame: A DataFrame containing the generated data.
    """
    
    # Generate independent variables (random floats between 0 and 1)
    data = np.random.rand(M, N)
    
    # Generate response variable
    if response_type == 'continuous':
        response = np.random.rand(M)  # Continuous values
    elif response_type == 'categorical':
        response = np.random.choice(['A', 'B', 'C'], size=M)  # Random categories
    else:
        raise ValueError("response_type must be either 'continuous' or 'categorical'")
    
    # Create DataFrame
    column_names = [f'X{i+1}' for i in range(N)] + ['Response']
    df = pd.DataFrame(np.column_stack((data, response)), columns=column_names)
    
    return df

def paper_data(M: int, N: int, response_type='continuous',low_var=True):
    """
    Generates a random dataset with M rows and N independent variables plus one response variable
    according to the paper's setup:
      y_i = log(1 + exp(6 * X_i1)) + ε,   ε ~ N(0, 20). (setting to 10 for testing) - set to 20 now. 
    
    Parameters:
    -----------
    M (int): Number of rows (observations).
    N (int): Number of independent variables.
    response_type (str): Type of response variable ('continuous' or 'categorical').
    
    Returns:
    --------
    pd.DataFrame: A DataFrame with N columns X1..XN and one column 'Response'.
    """
    
    # Generate N independent variables from Uniform(0,1)
    X = np.random.rand(M, N)
    
    if response_type == 'continuous':
        # eps ~ N(0, 20), i.e. standard deviation = sqrt(20)
        if low_var:
            eps = np.random.normal(0, 5, size=M)
        else:
            eps = np.random.normal(0, 20, size=M)
        
        # According to y_i = log(1 + exp(6 * X_i1)) + eps
        # (X_i1 is the first column of X, i.e., X[:, 0])
        response = np.log(1 + np.exp(6 * X[:, 0])) + eps
        
    elif response_type == 'categorical':
        # For illustration, generate random categories if desired
        response = np.random.choice(['A', 'B', 'C'], size=M)
    
    else:
        raise ValueError("response_type must be either 'continuous' or 'categorical'")

    # Combine into a single DataFrame
    column_names = [f'X{i+1}' for i in range(N)] + ['Response']
    df = pd.DataFrame(np.column_stack((X, response)), columns=column_names)
    
    return df


def generate_random_lin_table(M: int, N: int, response_type='continuous'):
    """
    Generates a random dataset with M rows and N independent variables plus one response variable,
    where the response variable is generated as a linear combination of the independent variables 
    with added Gaussian noise. This simulates data from a multiple linear regression model.
    
    Parameters:
        M (int): Number of rows (observations).
        N (int): Number of independent variables.
        response_type (str): Type of response variable; for this linear regression data 
                             only 'continuous' is supported.
                             
    Returns:
        pd.DataFrame: A DataFrame containing the generated data with columns X1, X2, ... XN and 'Response'.
    """
    
    if response_type != 'continuous':
        raise ValueError("For benchmarking multiple linear regression, response_type must be 'continuous'")
    
    # Generate independent variables (random floats between 0 and 1)
    X = np.random.rand(M, N)
    
    # Define deterministic coefficients for each independent variable.
    # Here we use a simple increasing sequence [1, 2, ..., N] for clarity.
    beta = np.arange(1, N + 1)
    
    # Generate response variable: a linear combination of the features plus Gaussian noise.
    noise = np.random.normal(loc=0, scale=0.1, size=M)
    response = X.dot(beta) + noise
    
    # Create DataFrame with feature columns and one response column
    column_names = [f'X{i+1}' for i in range(N)] + ['Response']
    df = pd.DataFrame(np.column_stack((X, response)), columns=column_names)
    
    return df


def paper_data2(M: int, N: int, response_type='continuous',low_var=True):
    """
    Generates dataset according to provided image:
    y = 10*sin(pi*X1*X2) + 20*(X3-0.5)^2 + 10*X4 + 5*X5 + eps
    with pi=5, eps ~ N(0, (5 or 20 depending on which is needed)).

    Parameters:
    -----------
    M (int): Number of rows (observations).
    N (int): Number of independent variables (should be at least 5).
    response_type (str): Only 'continuous' implemented.

    Returns:
    --------
    pd.DataFrame: DataFrame with columns X1..XN and 'Response'.
    """
    global is_low_var
    is_low_var = low_var
    if N < 5:
        raise ValueError("N must be at least 5 for the given equation.")

    X = np.random.rand(M, N)
    

    if response_type == 'continuous':
        if low_var:
            eps = np.random.normal(0, 5, size=M)
        else:
            eps = np.random.normal(0, 20, size=M)
        pi = 5
        response = (10 * np.sin(pi * X[:, 0] * X[:, 1]) +
                    20 * (X[:, 2] - 0.5)**2 +
                    10 * X[:, 3] +
                    5 * X[:, 4] + eps)
    else:
        raise ValueError("Only continuous response_type is implemented.")

    column_names = [f'X{i+1}' for i in range(N)] + ['Response']
    df = pd.DataFrame(np.column_stack((X, response)), columns=column_names)

    return df

def paper_data2_gamma(M: int,
                     N: int,
                     low_var: bool = True) -> pd.DataFrame:
    """
    Generates a synthetic dataset for Gamma regression:
        E[Y] = 10*sin(pi*X1*X2) + 20*(X3-0.5)^2 + 10*X4 + 5*X5

    Response Y is drawn from a Gamma distribution with
        shape = k,        scale = mu/k
    so that E[Y] = mu, Var[Y] = mu^2 / k.

    Parameters:
    -----------
    M : int
        Number of rows (observations).
    N : int
        Number of independent variables (must be >= 5).
    low_var : bool
        If True, uses shape k = 20 (lower dispersion).
        If False, uses shape k = 5  (higher dispersion).

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns X1..XN and 'Response'.
    """
    if N < 5:
        raise ValueError("N must be at least 5 for the given equation.")
    
    # 1) Draw predictors uniformly in [0,1]
    X = np.random.rand(M, N)
    
    # 2) Compute the mean function mu
    pi_val = np.pi
    mu = (
        10 * np.sin(pi_val * X[:, 0] * X[:, 1]) +
        20 * (X[:, 2] - 0.5)**2 +
        10 * X[:, 3] +
        5  * X[:, 4]
    )
    
    # Ensure mu > 0
    min_mu = mu.min()
    if min_mu <= 0:
        mu = mu - min_mu + 1e-6
    
    # 3) Draw from Gamma(shape=k, scale=mu/k)
    k = 20 if low_var else 5
    scale = mu / k
    response = np.random.gamma(shape=k, scale=scale, size=M)
    
    # 4) Assemble into DataFrame
    cols = [f"X{i+1}" for i in range(N)] + ["Response"]
    df = pd.DataFrame(
        np.column_stack([X, response]),
        columns=cols
    )
    return df

def paper_data2_poisson(M: int,
                        N: int) -> pd.DataFrame:
    """
    Generates a synthetic dataset for Poisson regression:
        E[Y] = 10*sin(pi*X1*X2) + 20*(X3-0.5)^2 + 10*X4 + 5*X5

    Response Y is drawn from a Poisson distribution with
        lambda = mu
    so that E[Y] = mu, Var[Y] = mu.

    Parameters:
    -----------
    M : int
        Number of rows (observations).
    N : int
        Number of independent variables (must be >= 5).

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns X1..XN and 'Response'.
    """
    if N < 5:
        raise ValueError("N must be at least 5 for the given equation.")
    
    # 1) Draw predictors uniformly in [0,1]
    X = np.random.rand(M, N)
    
    # 2) Compute the mean function mu
    pi_val = np.pi
    mu = (
        10 * np.sin(pi_val * X[:, 0] * X[:, 1]) +
        20 * (X[:, 2] - 0.5)**2 +
        10 * X[:, 3] +
        5  * X[:, 4]
    )
    
    # Ensure mu > 0 for the Poisson rate
    min_mu = mu.min()
    if min_mu <= 0:
        mu = mu - min_mu + 1e-6
    
    # 3) Draw from Poisson(lambda=mu)
    response = np.random.poisson(lam=mu, size=M)
    
    # 4) Assemble into DataFrame
    cols = [f"X{i+1}" for i in range(N)] + ["Response"]
    df = pd.DataFrame(
        np.column_stack([X, response]),
        columns=cols
    )
    return df


def compute_leaf_frequencies_vectorized(model, X_train, x):
    """
    Parameters
    ----------
    model : RandomForestRegressor
        A fitted random forest model from sklearn.
    X_train : ndarray of shape (n_samples, n_features)
        The training data used to fit the model.
    x : ndarray of shape (n_features,) or (1, n_features)
        A single new sample for which we want to compute frequencies.

    Returns
    -------
    alpha : ndarray of shape (n_samples,)
        alpha[i] = fraction of trees where X_train[i] ends up in the 
                   same leaf as x.
    """
    
    x_2d = np.array(x, ndmin=2)
    n_trees = len(model.estimators_)

    # shape: (n_trees, n_samples)
    train_leaf_indices_all = np.array([
        tree.apply(X_train) for tree in model.estimators_
    ])
    # shape: (n_trees,)
    x_leaf_indices = np.array([
        tree.apply(x_2d)[0] for tree in model.estimators_
    ])

    # Compare each row of train_leaf_indices_all to x_leaf_indices
    # => Boolean array of shape (n_trees, n_samples)
    same_leaf = (train_leaf_indices_all == x_leaf_indices[:, None])

    # sum across trees (axis=0) => shape (n_samples,)
    # then divide by n_trees
    alpha = same_leaf.sum(axis=0) / n_trees
    return alpha

hyperparam_names = ["forest_lambda","n_estimators","ridge_lambda","max_depth","min_leaf_size","omega"]
with open('best_params.json') as rf:
    best_params = json.load(rf)

def get_best_hyperparams(key,best_params=best_params):
    combo = eval(best_params[key]['best_param_combo'])
    return dict(zip(hyperparam_names, combo))

class LLTree():
    """
    build a single tree of many. Trees together build weights for the local regression. 
    """
    def __init__(self,X_vec,Y_vec,honest=True,max_depth = 4,min_leaf_size=10,ridge_lambda=1,omega=0.2,mode='gauss'):
        self.ridge_lambda = ridge_lambda
        self.max_depth = max_depth
        self.min_leaf_size=min_leaf_size
        self.omega = omega
        self.mode = mode
        if honest:
            # honest forest data splits
            idx = np.random.permutation(len(X_vec))
            half = len(idx) // 2
            self.Ix, self.Iy = X_vec.iloc[idx[:half]],  Y_vec.iloc[idx[:half]]
            self.Jx, self.Jy = X_vec.iloc[idx[half:]], Y_vec.iloc[idx[half:]]
            self.train_honest()
        else:
            self.Ix, self.Iy = X_vec,  Y_vec
            self.Jx, self.Jy = X_vec, Y_vec
            self.tree = build_regression_tree(self.Ix, self.Iy,max_depth=self.max_depth,min_leaf_size=self.min_leaf_size,ridge_lambda=self.ridge_lambda,omega=self.omega,mode=self.mode)
            self.tree_type = 'regular'
    
    def train_honest(self):
        """
        since we dont need to model strong smooth signals in the trees, limit recursion
        """
        tree = build_regression_tree(self.Ix, self.Iy,max_depth=self.max_depth,ridge_lambda=self.ridge_lambda,omega=self.omega,mode=self.mode)
        # honest forest - essentially have to run all the samples in the untouched data through the tree.
        # find indexes which rows are in that leaf, average them and set the new value
        X_copy = self.Jx.copy()
        X_copy['Y'] = self.Jy
        
        # TODO: rewrite from .apply to vectorized. 
        X_copy['leaf_id'] = X_copy.apply(lambda row: predict_single_sample(tree,row,output_id=True),axis=1)
        
        # if a particular tree leaf has samples assigned to it, average the response variable for those. 
        new_Y = X_copy.groupby(['leaf_id']).mean()[['Y']]
        new_tree = overwrite_tree_response(tree,new_Y)
        self.tree = new_tree
        self.tree_type = 'honest'
        
        
        
    def predict(self,sample):
        if len(sample.shape) == 1:
            return predict_single_sample(self.tree,sample)
        else:
            return sample.apply(lambda row: predict_single_sample(self.tree,row),axis=1)
        
class LLForest():
    """
    Implements random regression forest
    creates forest from n_estimators
    feature selection for individual learners happens in this class
    
    """
    def __init__(self,X_vec,Y_vec,n_estimators=10,ridge_lambda = 1,nr_feats = 5,max_depth = 5,min_leaf_size=10,omega=0.2,mode='gauss'):
        """
        nr feats = pi and d = total feats in data as per llf paper
        """
        # forest_lambda not actually necessary here, only used for local regression not in forest
        # self.forest_lambda = forest_lambda
        self.n_estimators = n_estimators
        self.ridge_lambda = ridge_lambda
        self.nr_feats = nr_feats
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.omega = omega
        self.X_vec = X_vec
        self.Y_vec = Y_vec
        self.mode = mode
        self.X_feats = list(self.X_vec.columns)
    
    def select_learner_features(self):
        """
        returns list of size nr_feats of randomly sampled feature names of the the entire feature name set 
        """
        return random.sample(self.X_feats,self.nr_feats)
        
        
    def train(self):
        """
        feature selection happens here
        """
        self.estimators = []
        for estimator_index in range(0,self.n_estimators):

            feats = self.select_learner_features()
            estimator = LLTree(self.X_vec[feats],self.Y_vec,ridge_lambda=self.ridge_lambda,max_depth=self.max_depth,omega=self.omega,mode=self.mode)
            self.estimators.append(estimator)
    
    def predict(self,single_sample):
        sum_estimates = 0
        for estimator in self.estimators:
            estimate = estimator.predict(single_sample)
            sum_estimates+=estimate
        return sum_estimates/self.n_estimators
        
    
    def calculate_forest_weights(self,x0):
        """
        """
        if not hasattr(self,'estimators'):
            print("Forest hasnt been trained. Training now...")
            self.train()
            self.calculate_forest_weights(x0)
        else:
            # empty the dataframe, leaving only the index
            local_X = self.X_vec.copy()
            local_X = local_X.drop(columns=local_X.columns)
            
            # create a column for each tree, based on test point x0.
            # 1 if the training set point falls within the same leaf as the probe point
            # whose neighborhood we are interested in modelling. else 0. 
            
            # have to normalize these weights by leaf size...in the training set?? 
            for estimator_number in range(0,self.n_estimators):
                
                estimator = self.estimators[estimator_number]
                x0_leaf = predict_single_sample(estimator.tree,x0,output_id=True)
                
                local_X[estimator_number] = self.X_vec.apply(lambda row: predict_single_sample(estimator.tree, row, output_id=True),axis=1)
                leaf_size = (local_X[estimator_number] == x0_leaf).astype(int).sum()
                local_X[estimator_number] = local_X[estimator_number] == x0_leaf
                local_X[estimator_number] = local_X[estimator_number]/leaf_size
            return local_X.sum(axis=1)/self.n_estimators


def make_llf_model(X, Y, mode='gauss', params=None):
    # pick & display at a random index
    randind = randrange(X.shape[0])
    print(f"chosen index: {randind} out of {X.shape[0]}")
    x0 = X.iloc[randind]

    #  default params to empty dict
    params = params or {}

    #  figure out which keys LLForest.__init__ actually takes
    sig = inspect.signature(LLForest.__init__)
    valid_forest_keys = set(sig.parameters) - {'self'}  # drop 'self'

    # partition
    forest_kwargs = {k: v for k, v in params.items() if k in valid_forest_keys}
    solver_kwargs = {k: v for k, v in params.items() if k not in valid_forest_keys}

    #  build + train forest
    forest_model = LLForest(X_vec=X, Y_vec=Y, mode=mode, **forest_kwargs)
    forest_model.train()

    #  compute weights & histogram
    forest_weights = forest_model.calculate_forest_weights(x0)
    forest_weights.hist(bins=30)

    # solve at x0, passing along any extra params (e.g. forest_lambda)
    mutest, thetatest = solve_llf_at_x0(
        x0, X, Y, forest_weights,
        **solver_kwargs
    )
    print(mutest, "", Y.iloc[randind])
    return forest_model, forest_weights

# jit placeholder
def batch_solve_llf(X,Y,forest_model,X_test,Y_test):
    Y_pred = []
    Y_pred_test = []
    length = X.shape[0]
    for row in range(0,length):
        if row == np.floor(length/2):
            print(row)
        x0 = X.iloc[row]
        alfa = forest_model.calculate_forest_weights(x0)
        mutest, thetatest = solve_llf_at_x0(x0,X,Y,alfa,forest_lambda=0.1)
        try:
            Y_pred.append(mutest[0])
        except:
            Y_pred.append(mutest)
    
    # repeat for test set
    length = X_test.shape[0]
    for row in range(0,length):
        if row == np.floor(length/2):
            print(row)
        x0 = X_test.iloc[row]
        alfa = forest_model.calculate_forest_weights(x0)
        mutest, thetatest = solve_llf_at_x0(x0,X,Y,alfa,forest_lambda=0.1)
        try:
            Y_pred_test.append(mutest[0])
        except:
            Y_pred_test.append(mutest)

    # make a random forest for benchmark
    # retrospektīvi, forcing same depth/nr estims as llf is a mistake 
    regr = RandomForestRegressor(max_depth=4,n_estimators=forest_model.n_estimators)
    regr.fit(X,Y)
    rf_pred = list(regr.predict(X))
    rf_pred_test = list(regr.predict(X_test))
    
    # make a xgboost model for benchmarks
    xgbregr = xgb.XGBRegressor(n_estimators=forest_model.n_estimators)
    xgbregr.fit(X,Y)
    xgb_pred = xgbregr.predict(X)
    xgb_pred_test = list(xgbregr.predict(X_test))

    models = {'rf':regr,'xgb':xgbregr}
    return Y_pred,Y_pred_test, xgb_pred, xgb_pred_test, rf_pred, rf_pred_test, models


# return Y_pred,Y_pred_test, xgb_pred, xgb_pred_test, rf_pred, rf_pred_test, models


# Y_train = reg_Y_train
# Y_test = reg_Y_test
# batch_solve_output = reg_ypred

def compare_models(Y_train, Y_test, batch_solve_output):
    # unpack predictions
    # llf_pred_train, llf_pred_test, \
    # xgb_pred_train, xgb_pred_test, \
    # rf_pred_train,  rf_pred_test = batch_solve_output

    llf_pred_train = batch_solve_output[0]
    llf_pred_test = batch_solve_output[1]
    xgb_pred_train = batch_solve_output[2]
    xgb_pred_test = batch_solve_output[3]
    rf_pred_train = batch_solve_output[4]
    rf_pred_test = batch_solve_output[5]

    fgr_train = plt.figure(figsize=(20, 16))
    plt.plot(llf_pred_train, label='LLF prediction', color='blue')
    y_train_obs = Y_train['Response'].to_numpy()
    plt.scatter(
        np.arange(len(y_train_obs)), 
        y_train_obs, 
        label='Treniņkopas atbildes mainīgais', 
        color='black'
    )
    plt.plot(xgb_pred_train, color='green', label='XGBRegressor')
    plt.plot(rf_pred_train,  color='orange', label='RandomForestRegressor')
    plt.legend()
    
    fgr_test = plt.figure(figsize=(20, 16))
    plt.plot(llf_pred_test, label='LLF prediction', color='blue')
    y_test_obs = Y_test['Response'].to_numpy()
    plt.scatter(
        np.arange(len(y_test_obs)), 
        y_test_obs, 
        label='Testa kopas atbildes mainīgais', 
        color='black'
    )
    plt.plot(xgb_pred_test, color='green',  label='XGBRegressor')
    plt.plot(rf_pred_test,  color='orange', label='RandomForestRegressor')
    plt.legend()


    # ---- TRAIN METRICS ----
    y_tr  = Y_train['Response'].to_numpy()
    n_tr  = len(y_tr)

    # RMSE for each model (train):
    llf_rmse_tr = np.sqrt( np.mean( (y_tr - llf_pred_train)**2 ) )
    xgb_rmse_tr = np.sqrt( np.mean( (y_tr - xgb_pred_train)**2 ) )
    rf_rmse_tr  = np.sqrt( np.mean( (y_tr - rf_pred_train)**2 ) )

    # R² for each model (train):
    llf_r2_tr = r2_score(y_tr, llf_pred_train)
    xgb_r2_tr = r2_score(y_tr, xgb_pred_train)
    rf_r2_tr  = r2_score(y_tr, rf_pred_train)

    # ---- TEST METRICS ----
    y_te  = Y_test['Response'].to_numpy()
    n_te  = len(y_te)

    llf_rmse_te = np.sqrt( np.mean( (y_te - llf_pred_test)**2 ) )
    xgb_rmse_te = np.sqrt( np.mean( (y_te - xgb_pred_test)**2 ) )
    rf_rmse_te  = np.sqrt( np.mean( (y_te - rf_pred_test)**2 ) )

    llf_r2_te = r2_score(y_te, llf_pred_test)
    xgb_r2_te = r2_score(y_te, xgb_pred_test)
    rf_r2_te  = r2_score(y_te, rf_pred_test)


    metrics_tr = {
        'LLF RMSE': llf_rmse_tr,
        'XGB RMSE': xgb_rmse_tr,
        'RF RMSE':  rf_rmse_tr,
        'LLF R2':   llf_r2_tr,
        'XGB R2':   xgb_r2_tr,
        'RF R2':    rf_r2_tr
    }

    metrics_te = {
        'LLF RMSE': llf_rmse_te,
        'XGB RMSE': xgb_rmse_te,
        'RF RMSE':  rf_rmse_te,
        'LLF R2':   llf_r2_te,
        'XGB R2':   xgb_r2_te,
        'RF R2':    rf_r2_te
    }

    summary = pd.DataFrame({'train': metrics_tr, 'test': metrics_te})
    return summary, fgr_train, fgr_test


def save_results(llf_model,batch_solve_output,X_train,Y_train,X_test,Y_test,dataset_name):
    os.chdir('/Users/skrs/Documents/uni/maģistrs/results')

    # location = os.getcwd()
    # location_path = location.split()[-2:]
    # correct_dir = location_path[-1] == 'results' and location_path[-2] == 'maģistrs'
    # if not correct_dir:
    #     try:
    #         os.chdir('/Users/skrs/Documents/uni/maģistrs/results')
    #     except:
    #         print(os.getcwd())
    #         raise Exception("???")
    #     except:
    #         try:
    #             os.chdir('maģistrs/results')
    #         except:
    #             try:
    #                 os.chdir('results')
    #             except:
    #                 raise Exception('bad directory location???')


    models = batch_solve_output[-1]
    models['llf'] = llf_model
    res, fgr_train, fgr_test = compare_models(Y_train,Y_test,batch_solve_output)

    # save the model outputs
    names = ['Y_pred_train', 'Y_pred_test', 'xgb_pred_train', 'xgb_pred_test', 'rf_pred_train', 'rf_pred_test']
    data_dict = dict(zip(names, batch_solve_output))

    os.chdir('data')
    for key in data_dict:
        fname = run_signature + '__' + dataset_name + '__'+ key + '.csv'
        pd.DataFrame({key:data_dict[key]}).to_csv(fname)
    os.chdir('..')

    # save the model inputs:
    names = ['X_train', 'X_test','Y_train','Y_test']
    data_dict = dict(zip(names, [X_train,X_test,Y_train,Y_test]))

    os.chdir('inputs')
    for key in data_dict.keys():
        fname = run_signature + '__' + dataset_name + '__' + key + '.csv'
        data_dict[key].to_csv(fname)
    os.chdir('..')


    # save the result table
    fname = run_signature + '__' + dataset_name + '__' + '.csv'
    res.to_csv(fname)


    # save the figures
    os.chdir('graphs')
    fname = run_signature + '__' + dataset_name + '__'+ 'train_figure'
    fname_pkl = fname + '.pkl'
    fname_img = fname + '.png'
    with open(fname_pkl,'wb') as wf:
        pickle.dump(fgr_train,wf)
    fgr_train.savefig(fname_img)

    fname = run_signature + '__' + dataset_name + '__'+ 'test_figure'
    fname_pkl = fname + '.pkl'
    fname_img = fname + '.png'
    with open(fname_pkl,'wb') as wf:
        pickle.dump(fgr_test,wf)
    fgr_test.savefig(fname_img)
    os.chdir('..')


    # save the models
    os.chdir('models')
    for key in models.keys():
        fname = run_signature + '__' + dataset_name + '__' + key + '.pkl'
        with open(fname, "wb") as f:
            pickle.dump(models[key], f)
    os.chdir('..')

def run_sims(M,N,run_signature):
    metadata = {'train_proportion':0.5}
    with open(os.getcwd() + '/metadata/' + run_signature + '.json','w+') as wf:
        json.dump(metadata,wf)
        
    # df = generate_random_table(M, N, response_type='continuous')
    # Y_vec = df[['Response']]
    # X_vec = df.drop(columns = ['Response'])
    
    # p_df = paper_data(M, N, response_type='continuous')
    # p_Y_vec = p_df[['Response']]
    # p_X_vec = p_df.drop(columns = ['Response'])

    # p_df_hvar = paper_data(M, N, response_type='continuous',low_var=False)
    # phvar_Y_vec = p_df_hvar[['Response']]
    # phvar_X_vec = p_df_hvar.drop(columns = ['Response'])

    p2_df = paper_data2(M, N, response_type='continuous')
    p2_Y_vec = p2_df[['Response']]
    p2_X_vec = p2_df.drop(columns=['Response'])

    p2hvar_df = paper_data2(M, N, response_type='continuous',low_var=False)
    p2hvar_Y_vec = p2hvar_df[['Response']]
    p2hvar_X_vec = p2hvar_df.drop(columns=['Response'])

    # p2g_df = paper_data2_gamma(M, N)
    # p2g_Y_vec = p2g_df[['Response']]
    # p2g_X_vec = p2g_df.drop(columns=['Response'])

    # p2g_df_hvar = paper_data2_gamma(M, N,low_var=False)
    # p2ghvar_Y_vec = p2g_df_hvar[['Response']]
    # p2ghvar_X_vec = p2g_df_hvar.drop(columns=['Response'])

    p2ps_df = paper_data2_poisson(M, N)
    p2ps_Y_vec = p2ps_df[['Response']]
    p2ps_X_vec = p2ps_df.drop(columns=['Response'])

    # lin_df = generate_random_lin_table(M, N, response_type='continuous')
    # lin_Y_vec = lin_df[['Response']]
    # lin_X_vec = lin_df.drop(columns=['Response'])


    # reg_X_train, reg_X_test, reg_Y_train, reg_Y_test = train_test_split(X_vec, Y_vec, test_size = metadata['train_proportion'])
    # reg_forest = make_llf_model(reg_X_train,reg_Y_train)

    # lin_X_train, lin_X_test, lin_Y_train, lin_Y_test = train_test_split(lin_X_vec, lin_Y_vec, test_size = metadata['train_proportion'])
    # lin_forest = make_llf_model(lin_X_train,lin_Y_train,params=get_best_hyperparams('lin'))

    # pub_X_train, pub_X_test, pub_Y_train, pub_Y_test = train_test_split(p_X_vec, p_Y_vec, test_size = metadata['train_proportion'])
    # pub_forest = make_llf_model(pub_X_train,pub_Y_train,params=get_best_hyperparams('pub1 low var'))

    # pub2_X_train, pub2_X_test, pub2_Y_train, pub2_Y_test = train_test_split(p2_X_vec, p2_Y_vec, test_size = metadata['train_proportion'])
    # pub2_forest = make_llf_model(pub2_X_train,pub2_Y_train,params=get_best_hyperparams('pub2 low var'))

    # lin likelihood split low var
    # pub2_X_train, pub2_X_test, pub2_Y_train, pub2_Y_test = train_test_split(p2_X_vec, p2_Y_vec, test_size = metadata['train_proportion'])
    # pub2_linsplit_forest = make_llf_model(pub2_X_train,pub2_Y_train,params=get_best_hyperparams('pub2, lin splits'))

    # pub_hvar_X_train, pub_hvar_X_test, pub_hvar_Y_train, pub_hvar_Y_test = train_test_split(phvar_X_vec, phvar_Y_vec, test_size = metadata['train_proportion'])
    # pub_hvar_forest = make_llf_model(pub_hvar_X_train,pub_hvar_Y_train,params=get_best_hyperparams('pub1 high var'))

    # pub2_hvar_X_train, pub2_hvar_X_test, pub2_hvar_Y_train, pub2_hvar_Y_test = train_test_split(p2hvar_X_vec, p2hvar_Y_vec, test_size = metadata['train_proportion'])
    # pub2_hvar_forest = make_llf_model(pub2_hvar_X_train,pub2_hvar_Y_train,params=get_best_hyperparams('pub2 high var'))

    # lin likelihood split high var
    # pub2_hvar_X_train, pub2_hvar_X_test, pub2_hvar_Y_train, pub2_hvar_Y_test = train_test_split(p2hvar_X_vec, p2hvar_Y_vec, test_size = metadata['train_proportion'])
    # pub2_hvar_linsplit_forest = make_llf_model(pub2_hvar_X_train,pub2_hvar_Y_train,params=get_best_hyperparams('hvar pub2, lin splits'))

    # add glm data
    pub2pois_X_train, pub2pois_X_test, pub2pois_Y_train, pub2pois_Y_test = train_test_split(p2ps_X_vec, p2ps_Y_vec, test_size = metadata['train_proportion'])
    pub2pois_pois_forest = make_llf_model(pub2pois_X_train,pub2pois_Y_train,params=get_best_hyperparams('pois pub2, pois splits'))
    pub2pois_lin_forest = make_llf_model(pub2pois_X_train,pub2pois_Y_train,params=get_best_hyperparams('pois pub2, lin splits'))
    pub2pois_forest = make_llf_model(pub2pois_X_train,pub2pois_Y_train,params=get_best_hyperparams('pois pub2, regular splits'))



    # pub = pub_forest[0]
    # lin = lin_forest[0]
    # pub2 = pub2_forest[0]
    # pubhvar = pub_hvar_forest[0]
    # pub2hvar = pub2_hvar_forest[0]
    # pub2 = pub2_linsplit_forest[0]
    # pub2hvar = pub2_hvar_linsplit_forest[0]
    pub2pois_pois = pub2pois_pois_forest[0]
    pub2pois_lin = pub2pois_lin_forest[0]
    pub2pois = pub2pois_forest[0]


    pub2pois_pois_ypred = batch_solve_llf(pub2pois_X_train,pub2pois_Y_train,pub2pois_pois,pub2pois_X_test, pub2pois_Y_test)
    pub2pois_lin_ypred = batch_solve_llf(pub2pois_X_train,pub2pois_Y_train,pub2pois_lin,pub2pois_X_test, pub2pois_Y_test)
    pub2pois_ypred = batch_solve_llf(pub2pois_X_train,pub2pois_Y_train,pub2pois,pub2pois_X_test, pub2pois_Y_test)
    # reg_ypred = batch_solve_llf(reg_X_train,reg_Y_train,reg, reg_X_test, reg_Y_test)
    # pub_ypred = batch_solve_llf(pub_X_train,pub_Y_train,pub,pub_X_test, pub_Y_test)
    # pub_hvar_ypred = batch_solve_llf(pub_hvar_X_train,pub_hvar_Y_train,pubhvar,pub_hvar_X_test, pub_hvar_Y_test)
    # pub2_hvar_ypred = batch_solve_llf(pub2_hvar_X_train,pub2_hvar_Y_train,pub2hvar,pub2_hvar_X_test, pub2_hvar_Y_test)
    # lin_ypred = batch_solve_llf(lin_X_train,lin_Y_train,lin,lin_X_test, lin_Y_test)
    # pub2_ypred = batch_solve_llf(pub2_X_train,pub2_Y_train,pub2,pub2_X_test, pub2_Y_test)


    # save_results(reg,reg_ypred,reg_X_train,reg_Y_train,reg_X_test,reg_Y_test,'iid X and Y by np.random.rand')
    # save_results(pub,pub_ypred,pub_X_train,pub_Y_train,pub_X_test,pub_Y_test,'eq. 1 in llf publication')
    # save_results(pubhvar,pub_hvar_ypred,pub_hvar_X_train,pub_hvar_Y_train,pub_hvar_X_test,pub_hvar_Y_test,'eq. 1 in llf publication with high variance')
    # save_results(pub2hvar,pub2_hvar_ypred,pub2_hvar_X_train,pub2_hvar_Y_train,pub2_hvar_X_test,pub2_hvar_Y_test,'eq. 7 in llf publication with high variance w llf splits')
    # save_results(pub2,pub2_ypred,pub2_X_train,pub2_Y_train,pub2_X_test,pub2_Y_test,'eq. 7 in llf publication w llf splits')
    # save_results(lin,lin_ypred,lin_X_train,lin_Y_train,lin_X_test,lin_Y_test,'Y = beta X + noise')
    save_results(pub2pois_pois,pub2pois_pois_ypred,pub2pois_X_train,pub2pois_Y_train,pub2pois_X_test,pub2pois_Y_test,'pois eq. 7 in llf publication with low variance w pois splits')
    save_results(pub2pois_lin,pub2pois_lin_ypred,pub2pois_X_train,pub2pois_Y_train,pub2pois_X_test,pub2pois_Y_test,'pois eq. 7 in llf publication with low variance w linear ll splits')
    save_results(pub2pois,pub2pois_ypred,pub2pois_X_train,pub2pois_Y_train,pub2pois_X_test,pub2pois_Y_test,'pois eq. 7 in llf publication with low variance w llf splits')


# for N in [10,20,30,40,50]:
#     M = 2000 
#     for i in range(0,8):
#         os.chdir('/Users/skrs/Documents/uni/maģistrs/results')
#         timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') 
#         run_signature = timestamp + '_with_' + str(N) + '_cols_' + str(M)+ '_rows'
#         run_sims(M,N,run_signature)
#         plt.close()