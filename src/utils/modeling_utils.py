import optuna
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import numpy as np

def tune_hyperparameters(X_train, y_train, X_val, y_val, train_penalty, config):
    """
    Tunes hyperparameters for a CatBoost model using Optuna.
    """
    tuning_config = config.get('hyperparameter_tuning', {})
    n_trials = tuning_config.get('n_trials', 50)

    def objective(trial):
        param = {
            'objective': 'Logloss',
            'eval_metric': 'AUC',
            'verbose': 0,
            'random_state': config['modeling']['random_state'],
            'depth': trial.suggest_int('depth', 
                                     tuning_config.get('depth_min', 4), 
                                     tuning_config.get('depth_max', 10)),
            'learning_rate': trial.suggest_float('learning_rate', 
                                               tuning_config.get('learning_rate_min', 0.01), 
                                               tuning_config.get('learning_rate_max', 0.3), log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 
                                             tuning_config.get('l2_leaf_reg_min', 1.0), 
                                             tuning_config.get('l2_leaf_reg_max', 10.0), log=True),
            'subsample': trial.suggest_float('subsample', 
                                           tuning_config.get('subsample_min', 0.6), 
                                           tuning_config.get('subsample_max', 1.0)),
            'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1,
        }

        # Apply penalty weights for CatBoost model
        penalty_weight = trial.suggest_float('penalty_weight', 
                                           tuning_config.get('penalty_weight_min', 2.0),
                                           tuning_config.get('penalty_weight_max', 10.0))
        
        train_weights = np.where(train_penalty == 1, penalty_weight, 1)

        model = CatBoostClassifier(**param)
        
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  sample_weight=train_weights,
                  early_stopping_rounds=tuning_config.get('early_stopping_rounds', 50),
                  verbose=0)
        
        preds = model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, preds)
        
        return auc_score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print(f"Best trial for {config['modeling']['model_types'][0]}:")
    print(f"  Value: {study.best_value}")
    print("  Params: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    return study.best_params
