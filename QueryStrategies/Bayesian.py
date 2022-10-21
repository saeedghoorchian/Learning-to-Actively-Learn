
import torch
import numpy as np


def max_entropy(learner, X, n_instances, T=100):

    random_subset = np.random.choice(range(len(X)), size=200, replace=False)
    with torch.no_grad():

        outputs = np.stack([learner(X[random_subset].float())[0].cpu().numpy() for t in range(100)])

    pc = outputs.mean(axis=0)

    acquisition = (-pc*np.log(pc + 1e-10)).sum(axis=-1)

    idx = (-acquisition).argsort()[:n_instances]
    query_idx = random_subset[idx]
    return query_idx, X[query_idx]


def bald(learner, X, n_instances, T=100):
    random_subset = np.random.choice(range(len(X)), size=200, replace=False) #2000
    with torch.no_grad():
        probs, _, _ = learner(X[random_subset].float())

        outputs = np.stack([learner(X[random_subset].float())[0].cpu().numpy() for t in range(100)])
 
    pc = outputs.mean(axis=0)
    H   = (-pc*np.log(pc + 1e-10)).sum(axis=-1)
    E_H = - np.mean(np.sum(outputs * np.log(outputs + 1e-10), axis=-1), axis=0)
    acquisition = H - E_H
    idx = (-acquisition).argsort()[:n_instances]
    query_idx = random_subset[idx]
    return query_idx, X[query_idx]

