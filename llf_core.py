import numpy as np
import pandas as pd 
from sklearn.linear_model import Ridge
from sklearn.linear_model import (
    LinearRegression,      # Gaussian / identity
    PoissonRegressor,      # Poisson / log
    GammaRegressor,        # Gamma / inverse
    TweedieRegressor       # Tweedie (power-family + link)
)

import statsmodels.api as sm
from numba import jit

# jit placeholder
def weighted_ridge_regression(X, y, alpha, forest_lambda):
    """
    Solve the weighted ridge regression problem:

        min_{mu, theta}
            (y - mu*1 - X*theta)^T W (y - mu*1 - X*theta) + lam * ||theta||^2

    where W = diag(alpha), i.e. alpha_i are the weights on each data point.

    We do *not* penalize mu (the intercept), only theta.

    Parameters
    ----------
    X : array-like, shape (n_samples, d_features)
        Design matrix (without intercept column).
    y : array-like, shape (n_samples,)
        Response values.
    alpha : array-like, shape (n_samples,)
        Weights for each sample (non-negative).
    lam : float
        Regularization strength (lambda).

    Returns
    -------
    mu : float
        The fitted intercept.
    theta : ndarray, shape (d_features,)
        The fitted coefficient vector.
    """
    # Convert to NumPy arrays
    X = np.asarray(X)
    y = np.asarray(y)
    alpha = np.asarray(alpha)

    n, d = X.shape
    
    # Diagonal weighting matrix
    # print("as_array alpha:")
    # print("shape:",alpha.shape)
    # print(alpha)

    W = np.diag(alpha)
    
    # Augment X with a column of ones for the intercept
    X_aug = np.column_stack([np.ones(n), X])  # shape (n, d+1)
    
    # Construct the un-regularized normal equations
    A = X_aug.T @ W @ X_aug  # shape (d+1, d+1)
    b = X_aug.T @ W @ y      # shape (d+1,)
    
    # Build the penalty matrix P for ridge:
    #  - no penalty on intercept => top-left entry = 0
    #  - penalty on theta => diagonal entries = 1 for the rest
    P = np.eye(d + 1)
    P[0, 0] = 0  # do not penalize the intercept
    
    # Add lambda * P to the normal equations
    A_reg = A + forest_lambda * P
    
    # Solve the system
    beta = np.linalg.solve(A_reg, b)
    
    # Extract mu and theta
    mu = beta[0]
    theta = beta[1:]
    return mu, theta

# jit placeholder
def solve_llf_at_x0(x0,X_in,Y_in,alphas_vec,forest_lambda=1):
    """
    take partial derivatives of the objective function, equate them to zero, 
    use some substitutions for the summands, 
    express theta_hat and mu_hat via these substitutions
    
    or just use the matrix notation
    """
    # print("alphas_vec")
    # print(alphas_vec)
    X = X_in - x0 
    mu_hat, theta_hat = weighted_ridge_regression(X,Y_in,alphas_vec,forest_lambda=forest_lambda)
    return mu_hat,theta_hat

# @jit(nopython=True)
def mean_loss(err_vec):
    return (err_vec**2).mean()


# @jit(nopython=True)
def log_likelihood(side_X,side_Y,mode,ridge_lambda,start_params=None):
    # https://www.statsmodels.org/stable/glm.html#module-reference
    # https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.html
    # https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.fit_regularized.html#statsmodels.genmod.generalized_linear_model.GLM.fit_regularized
    # https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLMResults.html#statsmodels.genmod.generalized_linear_model.GLMResults
    # whats the shape of side?
    # eksistē arī results.llf_scaled
    # šķiet, ka fit() neatļauj regularizēt, bet fit_regularized() neeksistē log-likelihood
    local_X = side_X
    local_Y = side_Y
    if mode == 'lin':
        model = sm.OLS(
            local_Y,
            local_X
        )
        try:
            results = model.fit(start_params=start_params)
        except ValueError:
            results = model.fit()
        return results.llf, results.params
    if mode == 'pois':
        model = sm.GLM(
            local_Y,
            local_X,
            family=sm.families.Poisson()
        )
        try:
            results = model.fit(start_params=start_params)
        except ValueError:
            results = model.fit()
        return results.llf, results.params
    elif mode == 'gamma':
        # Gamma GLM with L2 penalty (ridge_lambda)
        model = sm.GLM(
            local_Y,
            local_X,
            family=sm.families.Gamma()
        )
        try:
            results = model.fit(start_params=start_params)
        except ValueError:
            results = model.fit()
        return results.llf, results.params
    

    
# @jit(nopython=True)
def find_best_split(X_input,Y_input,omega=0.05,ridge_lambda=1.0,mode='gauss'):
    # get residuals of a linear regression
    local_df = pd.concat([X_input,Y_input],axis=1)
    
    # if mode == 'pois':
    #     local_df['Predicted'] = PoissonRegressor(alpha=ridge_lambda).fit(X_input,Y_input.values.reshape(-1, 1)).predict(X_input)
    # elif mode == 'gamma':
    #     local_df['Predicted'] = GammaRegressor(alpha=ridge_lambda).fit(X_input,Y_input.values.reshape(-1, 1)).predict(X_input)
    # else:
    #     local_df['Predicted'] = Ridge(alpha=ridge_lambda).fit(X_input,Y_input).predict(X_input)
    if mode == 'gauss':

        X_sm = sm.add_constant(X_input)
        model = sm.OLS(Y_input, X_sm)
        results = model.fit_regularized(alpha=ridge_lambda, L1_wt=0)
        local_df['Predicted'] = results.predict(X_sm)
        local_df['AbsErr'] = local_df['Response'] - local_df['Predicted']
    # set up variables for iterating
    M = X_input.shape[0]
    best_feature = X_input.T.index[0]
    least_total_loss = np.inf
    all_loss = []
    start_params_left = None 
    start_params_right = None

    # find the feature to split and where to minimize regression mean quadratic loss
    for feature in X_input:
        df_sorted_by_feature = local_df.sort_values(feature)
        for i in range(1,M):
            left = df_sorted_by_feature.iloc[0:i]
            right = df_sorted_by_feature.iloc[i:,]
            if i <= 1:
                best_left = left
                best_right = right
                best_feature = feature 
                breakpoint = left.iloc[i-1][feature]
            ls = left.shape[0]
            rs = right.shape[0]
            fraction = ls/(ls+rs)
            if fraction <= omega or fraction > 1- omega:
#                 print("fraction ",fraction," <= ",omega, "or fraction > ",1-omega)
                continue
            # for GLMs total_loss is replaced by sum of left and right log likelihoods times -1 as we look for the max. 
            if mode == 'gauss':
                left_loss = mean_loss(left['AbsErr'])
                right_loss = mean_loss(right['AbsErr'])
                total_loss = left_loss + right_loss
            else:
                # minus one because i dont want to rewrite the minimum finding. 
                left_X = left.drop(columns=['Response']).values
                left_Y = left['Response'].values
                right_X = right.drop(columns=['Response']).values
                right_Y = right['Response'].values
                try:
                    ll_left = log_likelihood(side_X=left_X,side_Y=left_Y,mode=mode,ridge_lambda=ridge_lambda,start_params=start_params_left)
                    ll_right = log_likelihood(side_X=right_X,side_Y=right_Y,mode=mode,ridge_lambda=ridge_lambda,start_params=start_params_right)
                except ValueError:
                    print(i," out of ",M)
                    continue
                start_params_left = ll_left[1]
                start_params_right = ll_right[1]
                total_loss = -1 * (ll_left[0] + ll_right[0])

            if total_loss < least_total_loss:
                all_loss.append(total_loss)
                best_feature = feature
                least_total_loss = total_loss
                breakpoint = left.iloc[i-1][feature]
                # best_split_row = left.index[-1]
                best_left, best_right = left,right
#     print("best feat: ", best_feature," breakpoint: ", breakpoint)
    if mode == 'gauss':
        return best_left.drop(columns=['AbsErr','Predicted','Response']),\
                best_left[['Response']],\
                best_right.drop(columns=['AbsErr','Predicted','Response']),\
                best_right[['Response']],\
                best_feature,\
                breakpoint
    else:
        return best_left.drop(columns=['Response']),\
                best_left[['Response']],\
                best_right.drop(columns=['Response']),\
                best_right[['Response']],\
                best_feature,\
                breakpoint


# jit placeholder
def uniqueness_not_ok(Y_sub):
    if (type(Y_sub.nunique()) == int):
        return Y_sub.nunique() == 1
    else:
        return Y_sub.nunique().iloc[0] == 1    

# jit placeholder
def build_regression_tree(X, Y, max_depth=5, min_leaf_size=10,ridge_lambda=1,omega=0.05,mode='gauss'):
    """
    Builds a regression tree by recursively finding the best split and creating child nodes.
    Each leaf node is assigned a unique ID ("leaf_id").
    Leaf node ids are necessary when building honest trees.
    """

    # This counter will be shared among all recursive calls to assign leaf IDs
    leaf_id_counter = 0

    # jit placeholder
    def build_tree(X_sub, Y_sub, depth,mode=mode):
        nonlocal leaf_id_counter  # so we can modify the counter from within this function

        def create_leaf_node(Y_values):
            nonlocal leaf_id_counter
            node = {
                "type": "leaf",
                "value": float(Y_values.mean().iloc[0]),  # Mean of Y in that node
                "leaf_id": leaf_id_counter               # Unique leaf ID
            }
            leaf_id_counter += 1
            return node

        # Stopping conditions
        if (
            depth <= 1
            or len(X_sub) <= min_leaf_size
            or uniqueness_not_ok(Y_sub)
                       
            ):
                return create_leaf_node(Y_sub)
            

        # Attempt to find the best split
        (
            best_left_X,
            best_left_Y,
            best_right_X,
            best_right_Y,
            best_feature,
            breakpoint_value,
        ) = find_best_split(X_sub, Y_sub,omega=omega,ridge_lambda = ridge_lambda,mode=mode)

        # If no valid split was found, make a leaf
        if (
            best_feature is None
            or best_left_X.empty
            or best_right_X.empty
        ):
            return create_leaf_node(Y_sub)

        # Otherwise, recurse on the left and right subsets
        left_subtree = build_tree(best_left_X, best_left_Y, depth - 1,mode=mode)
        right_subtree = build_tree(best_right_X, best_right_Y, depth - 1,mode=mode)

        # Construct internal node
        return {
            "type": "internal",
            "feature": best_feature,
            "threshold": breakpoint_value,
            "left": left_subtree,
            "right": right_subtree,
        }

    # Start recursion at the full dataset
    return build_tree(X, Y, max_depth)

# jit placeholder
def predict_single_sample(tree, sample,output_id=False):
        """
        Recursively traverse the tree for a single sample until a leaf is reached.

        Parameters
        ----------
        tree : dict
            A nested dictionary representing the regression tree.
        sample : pd.Series or dict-like
            A single row of features.

        Returns
        -------
        float
            The predicted value at the leaf.
        """

        if tree["type"] == "leaf":
            if output_id:
                return tree['leaf_id']
            else:
                return tree["value"]
        
        feature = tree["feature"]
        threshold = tree["threshold"]
        if sample[feature] <= threshold:
            return predict_single_sample(tree["left"], sample,output_id=output_id)
        else:
            return predict_single_sample(tree["right"], sample,output_id=output_id)

# jit placeholder    
def overwrite_tree_response(tree,new_Y):
    # assumes new_Y is a single column dataframe with column Y
    if tree['type'] == 'leaf':
        # leaf_id-indexed row in the new_Y dataframe has the new Y value 
        # however new_Y isnt guaranteed to have leaf_id-index. There might be no samples 
        # that fall within e.g. leaf 4 although the tree has leaf 4. 
        # in that case, dont overwrite. 
        if tree['leaf_id'] in new_Y.index.unique():
            tree['value'] = new_Y.loc[tree['leaf_id']]['Y']
        return tree
    else:
        overwrite_tree_response(tree['left'],new_Y)
        overwrite_tree_response(tree['right'],new_Y)
    return tree