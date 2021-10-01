from bayes_opt import BayesianOptimization
import xgboost as xgb

##### Bayesian Optimization of Hyper-parameters

def xgb_maximise(X_train, y_train):
    dtrain = xgb.DMatrix(X_train, label=y_train)

    def xgb_evaluate(learning_rate, max_depth, min_child_weight, gamma, subsample, colsample_bytree):
        params = {'eval_metric': 'error',
                  # 'scale_pos_weight': scale,
                  'learning_rate': learning_rate,
                  'max_depth': int(max_depth),
                  'min_child_weight': min_child_weight,
                  'subsample': subsample,
                  'eta': 0.1,
                  'gamma': gamma,
                  'colsample_bytree': colsample_bytree}
        cv_result = xgb.cv(params, dtrain, num_boost_round=100, nfold=10, metrics="aucpr")

        return cv_result['test-aucpr-mean'].iloc[-1]

    # BayesianOptimization will optimize the xgb_evaluate function over the supplied parameter space
    xgb_bo = BayesianOptimization(xgb_evaluate, {
        'learning_rate': (0.0, 0.9),
        'max_depth': (3, 11),
        'min_child_weight': (1, 7),
        'gamma': (0, 1),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.3, 0.9)},
        random_state=7777)

    xgb_bo.maximize(init_points=10, n_iter=20, acq='ei')

    # Returning best params
    params = xgb_bo.max['params']
    params['max_depth'] = round(params['max_depth'])

    return params
