import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.data_loader import load_raw
from src.modeling import (
    make_future_target, chronological_split, machine_group_split,
    build_features, train_baseline_ladder, LABEL_MAP
)

def compute_all_bootstrap_cis():
    df = load_raw()
    
    # 1. Horizon B
    df_hor_b = make_future_target(df, horizon_mode='B', H_min=30.0, tol_min=10.0)
    train_a, test_a = chronological_split(df_hor_b, train_ratio=0.8)
    Xa_train, scaler_a, km_a = build_features(train_a, is_model_a=True, fit_transforms=True)
    Xa_test, _, _ = build_features(test_a, is_model_a=True, scaler=scaler_a, kmeans_model=km_a, fit_transforms=False)
    ya_train, ya_test = train_a['target_future_code'].values, test_a['target_future_code'].values
    ladder_a, models_a, _ = train_baseline_ladder(Xa_train, ya_train, Xa_test, ya_test, task_name='Model A (Horizon B)')
    
    # 2. Horizon A
    df_hor_a = make_future_target(df, horizon_mode='A', N_shift=1)
    train_a_sens, test_a_sens = chronological_split(df_hor_a, train_ratio=0.8)
    Xa_train_sens, scaler_sens, km_sens = build_features(train_a_sens, is_model_a=True, fit_transforms=True)
    Xa_test_sens, _, _ = build_features(test_a_sens, is_model_a=True, scaler=scaler_sens, kmeans_model=km_sens, fit_transforms=False)
    ya_train_sens, ya_test_sens = train_a_sens['target_future_code'].values, test_a_sens['target_future_code'].values
    ladder_a_sens, models_a_sens, _ = train_baseline_ladder(Xa_train_sens, ya_train_sens, Xa_test_sens, ya_test_sens, task_name='Model A (Horizon A)')

    # 3. Model B (Diagnosis)
    df_diag = df.copy()
    df_diag['current_status_code'] = df_diag['Efficiency_Status'].map(LABEL_MAP)
    train_b, test_b = chronological_split(df_diag, train_ratio=0.8)
    Xb_train, scaler_b, km_b = build_features(train_b, is_model_a=False, fit_transforms=True)
    Xb_test, _, _ = build_features(test_b, is_model_a=False, scaler=scaler_b, kmeans_model=km_b, fit_transforms=False)
    yb_train, yb_test = train_b['current_status_code'].values, test_b['current_status_code'].values
    ladder_b, models_b, _ = train_baseline_ladder(Xb_train, yb_train, Xb_test, yb_test, task_name='Model B (Diagnosis)')

    # 4. Unseen Machines (Horizon B)
    train_grp, test_grp = machine_group_split(df_hor_b, test_machine_ratio=0.2)
    Xgrp_train, scaler_grp, km_grp = build_features(train_grp, is_model_a=True, fit_transforms=True, include_machine_id=False)
    Xgrp_test, _, _ = build_features(test_grp, is_model_a=True, scaler=scaler_grp, kmeans_model=km_grp, fit_transforms=False, include_machine_id=False)
    ygrp_train, ygrp_test = train_grp['target_future_code'].values, test_grp['target_future_code'].values
    ladder_grp, models_grp, _ = train_baseline_ladder(Xgrp_train, ygrp_train, Xgrp_test, ygrp_test, task_name='Model A (Unseen Machines)')

    def get_bootstrap_ci(model, X, y, B=1000, seed=42):
        np.random.seed(seed)
        N = len(y)
        try:
            probs = model.predict_proba(X)
        except Exception:
            return (0.5000, 0.5000)
        aucs = []
        for _ in range(B):
            idx = np.random.choice(N, size=N, replace=True)
            y_b = y[idx]
            if len(np.unique(y_b)) < 3:
                continue
            try:
                auc = roc_auc_score(y_b, probs[idx], multi_class='ovr', average='macro')
                aucs.append(auc)
            except Exception:
                pass
        if len(aucs) == 0:
            return (0.5000, 0.5000)
        return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))

    suites = [
        ('Model A (Horizon B)', models_a, Xa_test, ya_test, ladder_a),
        ('Model A (Horizon A)', models_a_sens, Xa_test_sens, ya_test_sens, ladder_a_sens),
        ('Model B (Diagnosis)', models_b, Xb_test, yb_test, ladder_b),
        ('Model A (Unseen Machines)', models_grp, Xgrp_test, ygrp_test, ladder_grp),
    ]

    name_map = {
        'Dummy': 'Dummy',
        'LogisticRegression': 'Logistic',
        'RandomForest': 'Random Forest',
        'LightGBM': 'LightGBM'
    }

    print('\n================ TABLE VII FULL METRICS WITH BOOTSTRAP CIs ================')
    for suite_name, m_dict, X_eval, y_eval, l_df in suites:
        print(f'\n--- {suite_name} ---')
        for m_key in ['Dummy', 'LogisticRegression', 'RandomForest', 'LightGBM']:
            if m_key not in m_dict:
                continue
            model = m_dict[m_key]
            ci_low, ci_high = get_bootstrap_ci(model, X_eval, y_eval)
            pattern = name_map[m_key]
            matches = l_df[l_df['Model'].str.contains(pattern, case=False)]
            if len(matches) == 0:
                continue
            row = matches.iloc[0]
            acc = row['Accuracy']
            mf1 = row['Macro_F1']
            wf1 = row['Weighted_F1']
            auc = row['ROC_AUC_Macro']
            p_high = row['Precision_High']
            r_high = row['Recall_High']
            print(f'{m_key:<18} | Acc: {acc:.4f} | MF1: {mf1:.4f} | WF1: {wf1:.4f} | AUC: {auc:.4f} [{ci_low:.4f}, {ci_high:.4f}] | High P: {p_high:.4f} | High R: {r_high:.4f}')

if __name__ == '__main__':
    compute_all_bootstrap_cis()
