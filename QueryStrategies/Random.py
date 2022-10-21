

import numpy as np


def uniform_random(learner, X, n_instances=1): 
    query_idx = np.random.choice(range(len(X)), size=n_instances, replace=False) #pool based!
    return query_idx, X[query_idx]

