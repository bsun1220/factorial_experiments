import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import cvxpy as cp
import itertools
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class FactorialWeights:

    def __init__(self):
        pass

    def preprocess_data(self, df, cov_cols, treat_cols, outcome, outlier_cutoff = 3.0):
        """
        does z-score normalization of non-binary data and MICE for missing data

        Parameters
        ----------
        cov_cols : list
            list of cov cols
        treat_cols : list
            list of treatment columns
        outcome : string
            outcome column name
        outlier_cutoff : double, optional
            cutoff to replace all values, default is 3.0, to 0 after z-score normalization
        """
        df1 = df.copy()

        for cov in cov_cols:
            df1[cov] = df[cov]
            if len(np.unique(df[cov])) > 2:
                df1[cov] = (df1[cov] - df1[cov].mean())/df1[cov].std()
                df1.loc[df1[cov].abs() > outerlier_cutoff, cov] = 0
        
        imp = IterativeImputer(max_iter=1000, random_state=0)
        imp.fit(df1[cov_cols])
        df1[cov_cols] = imp.transform(df1[cov_cols])

        for treat in treat_cols:
            df1[treat] = df[treat]
            df1[treat] = df1[treat].replace(0, -1)
            assert len(np.unique(df1[treat])) == 2
        df1[outcome] = df[outcome]
        return df1

    def calc_weighting(self, df, treat_cols, cov_cols, effects, weighting = 'entropy', intr_effects = 1, is_hetero = True):
        """
        performs balancing weights algorithm from https://arxiv.org/abs/2310.04660

        Parameters
        ----------
        df : pd.DataFrame
            dataframe containing outcome, treatment_columns, and covariates
        treat_cols : list
            list of treatment columns
        cov_cols : list
            list of cov cols
        effects : list
            list of treatments we want to measure the effects of
        weighting: string, optional
            possible options are 'entropy' and 'variance', indicating whether
            we want to minimize sum w_i log w_i (entropy) or sum w_i^2 (variance)
        intr_effects : int, optional
            variable for how many interactions we believe are important
            to include in the model (default is 1)
        is_hetero : boolean, optional
            assumes either model with treatment heterogeneity
            or general additive outcome model (default is True)
        """
        
        if is_hetero:
            return self.heterogeneous_treat_model(df, treat_cols, cov_cols, effects, weighting, intr_effects)
        else:
            return self.additive_treat_model(df, treat_cols, cov_cols, effects, weighting, intr_effects)

    def additive_treat_model(self, df, treat_cols, cov_cols, effects, weighting = 'entropy', intr_effects = 1):
        assert weighting == 'entropy' or weighting == 'variance'
        assert len(np.unique(df[treat_cols].values, axis = 0)) == 2**len(treat_cols)
        
        x = cp.Variable(len(df))
        obj = None
        if weighting == 'entropy':
            obj = cp.Minimize(-cp.sum(cp.entr(x)))
        if weighting == 'variance':
            obj = cp.Minimize(cp.sum_squares(x))
        
        constraints = [x >= 0]
        
        df = df.copy()
    
        df['g'] = df[effects].prod(axis = 1)
        
        df['A_plus'] = np.maximum(df['g'], 0)
        df['A_minus'] = np.maximum(-df['g'], 0)

        for cov in cov_cols:
            right_val = df[cov].sum()
            left_val_plus = df['A_plus'] * df[cov].sum()
            left_val_minus = df['A_minus'] * df[cov].sum()
            constraints.append(cp.multiply(x, left_val_plus).sum() == right_val)
            constraints.append(cp.multiply(x, left_val_minus).sum() == right_val)

        treat_combos = [[ele] for ele in treat_cols]
        for i in range(2, intr_effects + 1):
            vals = list(itertools.combinations(treat_cols, i))
            list_vals = [list(ele) for ele in vals]
            treat_combos.extend(list_vals)
        
        z_combos = pd.DataFrame(np.unique(df[treat_cols].values, axis = 0), columns = treat_cols)

        for treat in treat_combos:
            z_val = df[treat].prod(axis = 1)
            plus_const = df['A_plus'] * z_val * 1/len(df)
            minus_const = df['A_minus'] * z_val * 1/len(df)
            
            plus_res, minus_res = 0, 0
            for j in range(len(z_combos)):
                g_value = z_combos.loc[j][effects].prod()
                mult = z_combos.loc[j][treat].prod()
                if g_value > 0:
                    plus_res += mult
                if g_value < 0:
                    minus_res += mult
            
            plus_res /= (2**(len(treat_cols) - 1))
            minus_res /= (2**(len(treat_cols) - 1))
            constraints.append(cp.multiply(x, plus_const).sum() == plus_res)
            constraints.append(cp.multiply(x, minus_const).sum() == minus_res)
                
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.CLARABEL)
        return x    
            
    def heterogeneous_treat_model(self, df, treat_cols, cov_cols, effects, weighting = 'entropy', intr_effects = 1):
        assert weighting == 'entropy' or weighting == 'variance'
        assert len(np.unique(df[treat_cols].values, axis = 0)) == 2**len(treat_cols)
        
        x = cp.Variable(len(df))
        obj = None
        if weighting == 'entropy':
            obj = cp.Minimize(-cp.sum(cp.entr(x)))
        if weighting == 'variance':
            obj = cp.Minimize(cp.sum_squares(x))
        
        constraints = [x >= 0]
        
        df = df.copy()
    
        df['g'] = df[effects].prod(axis = 1)
        
        df['A_plus'] = np.maximum(df['g'], 0)
        df['A_minus'] = np.maximum(-df['g'], 0)
        
        treat_combos = [[ele] for ele in treat_cols]
        for i in range(2, intr_effects + 1):
            vals = list(itertools.combinations(treat_cols, i))
            list_vals = [list(ele) for ele in vals]
            treat_combos.extend(list_vals)
        
        sums = {}
        
        for cov_col in cov_cols:
            sums[cov_col] = df[cov_col].sum()
        
        #All Unique Z Combos
        z_combos = pd.DataFrame(np.unique(df[treat_cols].values, axis = 0), columns = treat_cols)
    
        for s in range(len(cov_cols)):
            for k in range(len(treat_combos)):
                cov_col = cov_cols[s]
                specific_treat_cols = treat_combos[k]
                
                #Get Treatment Vals
                z_val = df[specific_treat_cols].prod(axis = 1)
                
                #Left Side of Constraint
                plus_const = df['A_plus'] * df[cov_col] * z_val
                minus_const = df['A_minus'] * df[cov_col] * z_val
                
                #Right Side of Constraint
                plus_res, minus_res = 0, 0
                for j in range(len(z_combos)):
                    g_value = z_combos.loc[j][effects].prod()
                    mult = z_combos.loc[j][specific_treat_cols].prod()
                    if g_value > 0:
                        plus_res += sums[cov_col] * mult
                    if g_value < 0:
                        minus_res += sums[cov_col] * mult
                
                plus_res /= (2**(len(treat_cols) - 1))
                minus_res /= (2**(len(treat_cols) - 1))
                constraints.append(cp.multiply(x, plus_const).sum() == plus_res)
                constraints.append(cp.multiply(x, minus_const).sum() == minus_res)
    
        prob = cp.Problem(obj, constraints)
        prob.solve(verbose = False, solver=cp.CLARABEL)
        return x





                              