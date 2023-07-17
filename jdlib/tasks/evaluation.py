from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn import metrics

import torch

from jdlib.tasks import eval_protocols


def run_experiment_single_label(train_repr, train_labels, test_repr, test_labels, 
                                target_attr, eval_protocol="linear"):
    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    else:
        assert False, 'unknown evaluation protocol'
    
    clf = fit_clf(train_repr, train_labels)
    # acc = clf.score(test_repr, test_labels)
    
    if eval_protocol == 'linear':
        y_score = clf.predict_proba(test_repr)[:, 1]
    else:
        y_score = clf.decision_function(test_repr)
    
    return y_score


def evaluate_single_label(y_score, labels):
    labels_onehot = label_binarize(labels, classes=np.arange(2))
    auprc = metrics.average_precision_score(labels_onehot, y_score)
    auroc = metrics.roc_auc_score(labels_onehot, y_score)
    
    th = 0.5
    y_pred = (y_score > th).astype(int) if y_score.ndim==1 else np.argmax(y_score, axis=1)
    cm = metrics.confusion_matrix(labels, y_pred)
    
    return {"auprc": auprc, "auroc": auroc, "cm": {"TN": int(cm[0, 0]), 
                                                   "FP": int(cm[0, 1]), 
                                                   "FN": int(cm[1, 0]), 
                                                   "TP": int(cm[1, 1])}}


def run_classification(model, train_loader, test_loader, used_feature="both"):
    # producing representations
    train_g_repr, train_slots = model.encode(dataloader=train_loader)
    test_g_repr, test_slots = model.encode(dataloader=test_loader)
    
    # (B, D) -> (B, 1, D)
    if len(train_g_repr.shape) == 2:
        train_g_repr = train_g_repr.unsqueeze(1)
    if len(test_g_repr.shape) == 2:
        test_g_repr = test_g_repr.unsqueeze(1)
    
    if isinstance(used_feature, str):
        used_list = [used_feature]  
        return_nest = False
    else:
        used_list = used_feature
        return_nest = True
        
    all_results = {}
    for used_feature in used_list:
        assert used_feature in ["both", "global", "local"]
        if used_feature=="both":
            train_repr = torch.cat([train_g_repr.mean(axis=1), train_slots.mean(axis=1)], axis=1)
            test_repr = torch.cat([test_g_repr.mean(axis=1), test_slots.mean(axis=1)], axis=1)
        elif used_feature=="global":
            train_repr = train_g_repr.mean(axis=1)
            test_repr = test_g_repr.mean(axis=1)
        else:
            train_repr = train_slots.mean(axis=1)
            test_repr = test_slots.mean(axis=1)

        results = {}
        for target_attr in tqdm(train_loader.dataset.ecg_label.columns, desc="Downstream Evaluation: "):    
            # label preparation
            train_labels = train_loader.dataset.ecg_label[target_attr].values
            test_labels = test_loader.dataset.ecg_label[target_attr].values

            # classifier
            y_score = run_experiment_single_label(train_repr, train_labels, 
                                                  test_repr, test_labels, 
                                                  target_attr, eval_protocol="linear")

            result_dict = evaluate_single_label(y_score, test_labels)

            # store the experimental results
            results[target_attr] = result_dict

        # if multi-label linear regression
        # input: reprs in the shape of (B, n_repr, _D)
        # output: result dictionary
        all_results[used_feature] = results
    
    if return_nest:
        return all_results
    else:
        return results