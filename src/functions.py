# Imports
import pandas as pd

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate

from hyperopt import STATUS_OK

from scipy.stats import chi2_contingency

import seaborn as sns

def calculate_chi_square(contingency_table):
    # Returns chi-square statistic
    # takes the contingency table between the two features as argument
    
    chi2_stat, p, dof, expected = chi2_contingency(contingency_table)
    return chi2_stat

def calculate_cramers_v(chi2_stat, contingency_table):
    # Returns cramers_v measurement for the chi-square test of independence
    # takes chi-square statistic and contingency table between the two features as arguments
    # n number of observations
    # cat_ncount_first and cat_count_second are counts of unique categories in the two features
    
    n = contingency_table.sum().sum() 
    cat_count_first, cat_count_second = contingency_table.shape
    return np.sqrt(chi2_stat / (n * (min(cat_count_first - 1, cat_count_second - 1))))

def construct_cramers_v_matrix(data):
    # Returns Cramér's V for all pairs of categorical variables as a dataframe
    # the function takes dataframe as an argument
    # It creates a contingency table for each pair of variables, 
    # calculates chi-square and cramers_v and fills the resulted dataframe with the values symmetricaly
    columns = data.columns
    n = len(columns)
    result_matrix = pd.DataFrame(np.zeros((n, n)), index = columns, columns = columns)

    for i in range(n):
        for j in range(i, n):
            if i == j:
                result_matrix.iloc[i, j] = 1.0
            else:
                contingency_table = pd.crosstab(data.iloc[:, i], data.iloc[:, j])
                chi2_stat = calculate_chi_square(contingency_table)
                cramers_v = calculate_cramers_v(chi2_stat, contingency_table)
                result_matrix.iloc[i, j] = cramers_v
                result_matrix.iloc[j, i] = cramers_v

    return result_matrix

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


def hyperopt_objective(classifier, params, attr_train, target_train, eval_metrics, encoder, random_seed):
    clf = None
    clf_params = None
    if classifier == 'LogisticRegression':
        clf = LogisticRegression(solver = 'liblinear', random_state = random_seed)
        clf_params = {'classifier__C': params['C'], 'classifier__penalty': params['penalty']}
    else:
        raise Exception('The classifier must be \'LogisticRegression\', \'RandomForest\' or ...')
        
    pipeline = Pipeline(steps = [
            ('encode', encoder),
            ('classifier', clf)
        ]
    )

    pipeline.set_params(**clf_params)
    
    skf = StratifiedKFold(n_splits = 5)
    
    scores = cross_validate(pipeline, attr_train, target_train, cv = skf, scoring = eval_metrics)

    return {
        'loss': -np.mean(scores['test_recall']),
        'status': STATUS_OK,
        'other_metrics': {
            'mean_recall': np.mean(scores['test_recall']),
            'std_recall': np.std(scores['test_recall']),
            'mean_accuracy': np.mean(scores['test_accuracy']),
            'std_accuracy': np.std(scores['test_accuracy']),
            'mean_roc_auc': np.mean(scores['test_roc_auc']),
            'std_roc_auc': np.std(scores['test_roc_auc']),
        }
    }
