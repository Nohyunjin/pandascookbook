def bayes_parameter_opt_lgb(X, y, init_round=15, opt_round=25, n_folds=5, random_seed=6, n_estimators=10000, learning_rate=0.05, output_process=False):
# prepare data
    train_data = lgb.Dataset(data=X, label=y, free_raw_data=False)
# parameters
    def lgb_eval(num_leaves, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight, colsample_bytree):
        params = {'application':'binary','num_iterations': n_estimators, 'learning_rate':learning_rate, 'early_stopping_round':100, 'metric':'auc'}
        params["num_leaves"] = int(round(num_leaves))
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        params['colsample_bytree'] = colsample_bytree
        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=['auc'])
        return max(cv_result['auc-mean'])

    # range
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (24, 40), # 성능 향상
                                            'max_depth': (5, 8), # 성능 향상
                                            'lambda_l1': (0, 5), # 과적합 방지
                                            'lambda_l2': (0, 3), # 과적합 방지
                                            'min_split_gain': (0.01, 0.1), # 과적합 방지
                                            'min_child_weight': (5, 50), # 과적합 방지
                                            'bagging_fraction': (0.5, 1), # 속도 향상
                                            'colsample_bytree': (0.6, 1)}, # 속도 향상
                                            random_state=0)
    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
