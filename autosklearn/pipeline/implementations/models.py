# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
Original version was forked from https://github.com/ArdalanM/pyLightGBM
"""
from __future__ import print_function
import os
import re
import shutil
import tempfile
import subprocess
import numpy as np
import scipy.sparse as sps
import autosklearn.pipeline.implementations.io_utils as io_utils
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class GenericGBM(BaseEstimator):
    def __init__(self, exec_path="lightgbm",
                 config="",
                 application="regression",
                 n_estimators=10,
                 learning_rate=0.1,
                 num_leaves=127,
                 tree_learner="serial",
                 num_threads=1,
                 min_data_in_leaf=100,
                 metric='l2,',
                 is_training_metric=False,
                 feature_fraction=1.,
                 feature_fraction_seed=2,
                 bagging_fraction=1.,
                 bagging_freq=0,
                 bagging_seed=3,
                 metric_freq=1,
                 early_stopping_round=0,
                 max_bin=255,
                 is_unbalance=False,
                 num_class=1,
                 boosting_type='gbdt',
                 min_sum_hessian_in_leaf=10,
                 drop_rate=0.01,
                 drop_seed=4,
                 max_depth=-1,
                 lambda_l1=0.,
                 lambda_l2=0.,
                 min_gain_to_split=0.,
                 verbose=True,
                 model=None,
                 # autosklearn需要random_state
                 random_state=1,
                 train_file: str = None):

        # 默认使用路径中的lightgbm
        try:
            # 如果环境中有路径，则使用
            self.exec_path = os.environ['LIGHTGBM_EXEC']
        except KeyError:
            print(
                "pyLightGBM is looking for 'LIGHTGBM_EXEC' environment variable, cannot be found.")
            print(
                "exec_path will be deprecated in favor of environment variable")
            # 使用给出的路径，并扩展个人目录的绝对路径
            # '~/path/to/lightgbm' becomes 'absolute/path/to/lightgbm'
            self.exec_path = os.path.expanduser(exec_path)

        self.config = config
        self.model = model
        self.verbose = verbose
        # 为了让autosklearn调用，改名
        self.n_estimators = n_estimators
        # 如果使用train_file，则使用其绝对路径
        self.train_file = os.path.abspath(train_file)
        self.param = {
            'application': application,
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'tree_learner': tree_learner,
            'num_threads': num_threads,
            'min_data_in_leaf': min_data_in_leaf,
            'metric': metric,
            'is_training_metric': is_training_metric,
            'feature_fraction': feature_fraction,
            'feature_fraction_seed': feature_fraction_seed,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'bagging_seed': bagging_seed,
            'metric_freq': metric_freq,
            'early_stopping_round': early_stopping_round,
            'max_bin': max_bin,
            'is_unbalance': is_unbalance,
            'num_class': num_class,
            'boosting_type': boosting_type,
            'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,
            'drop_rate': drop_rate,
            'drop_seed': drop_seed,
            'max_depth': max_depth,
            'lambda_l1': lambda_l1,
            'lambda_l2': lambda_l2,
            'min_gain_to_split': min_gain_to_split,
            'random_state': random_state,
        }

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            test_data=None,
            init_scores=[]):
        """Base train function for both classificaiton and regression of the
        LightGBM.

        Parameters
        ----------
        X : array_like
            Features of the input.
        y : array_like
            Labels of the input.
        test_data : dict[X, y]
            A set of (train, test) pairs.
        init_scores
        """
        # create tmp dir to hold data and model (especially the latter)
        tmp_dir = tempfile.mkdtemp()
        issparse = sps.issparse(X)
        f_format = "svm" if issparse else "csv"

        # 如果指定了训练集，使用
        if self.train_file:
            train_filepath = self.train_file
        else:
            train_filepath = os.path.abspath(
                "{}/X.{}".format(tmp_dir, f_format))
            io_utils.dump_data(X, y, train_filepath, issparse)
        init_filepath = train_filepath + ".init"

        if len(init_scores) > 0:
            assert len(init_scores) == X.shape[0]
            np.savetxt(init_filepath, X=init_scores, delimiter=',',
                       newline=os.linesep)
        # else:
        #     if self.param['application'] in ['binary', 'multiclass']:
        #         np.savetxt(init_filepath, X=0.5 * np.ones(X.shape[0]),
        #                    delimiter=',', newline=os.linesep)

        if test_data:
            valid = []
            for i, (x_test, y_test) in enumerate(test_data):
                test_filepath = os.path.abspath(
                    "{}/X{}_test.{}".format(tmp_dir, i, f_format))
                valid.append(test_filepath)
                io_utils.dump_data(x_test, y_test, test_filepath, issparse)
            self.param['valid'] = ",".join(valid)

        self.param['task'] = 'train'
        self.param['data'] = train_filepath
        self.param['output_model'] = os.path.join(tmp_dir, "LightGBM_model.txt")

        calls = ["{}={}\n".format(k, self.param[k]) for k in self.param]

        if self.config == "":
            conf_filepath = os.path.join(tmp_dir, "train.conf")
            with open(conf_filepath, 'w') as f:
                f.writelines(calls)

            process = subprocess.Popen(
                [self.exec_path, "config={}".format(conf_filepath)],
                stdout=subprocess.PIPE, bufsize=1)

        else:
            process = subprocess.Popen(
                [self.exec_path, "config={}".format(self.config)],
                stdout=subprocess.PIPE, bufsize=1)

        with process.stdout:
            for line in iter(process.stdout.readline, b''):
                print(line.strip().decode('utf-8')) if self.verbose else None
        # wait for the subprocess to exit
        process.wait()

        with open(self.param['output_model'], mode='r') as file:
            self.model = file.read()
        shutil.rmtree(tmp_dir)

        if test_data and self.param['early_stopping_round'] > 0:
            self.best_round = max(
                map(int, re.findall("Tree=(\d+)", self.model))) + 1

    def predict(self, X):
        tmp_dir = tempfile.mkdtemp()
        issparse = sps.issparse(X)
        f_format = "svm" if issparse else "csv"

        predict_filepath = os.path.abspath(
            os.path.join(tmp_dir, "X_to_pred.{}".format(f_format)))
        output_model = os.path.abspath(os.path.join(tmp_dir, "model"))
        output_results = os.path.abspath(
            os.path.join(tmp_dir, "LightGBM_predict_result.txt"))
        conf_filepath = os.path.join(tmp_dir, "predict.conf")

        with open(output_model, mode="w") as file:
            file.write(self.model)

        io_utils.dump_data(X, np.zeros(X.shape[0]), predict_filepath, issparse)

        calls = ["task = predict\n",
                 "data = {}\n".format(predict_filepath),
                 "input_model = {}\n".format(output_model),
                 "output_result={}\n".format(output_results)]

        with open(conf_filepath, 'w') as f:
            f.writelines(calls)

        process = subprocess.Popen(
            [self.exec_path, "config={}".format(conf_filepath)],
            stdout=subprocess.PIPE, bufsize=1)

        with process.stdout:
            for line in iter(process.stdout.readline, b''):
                print(line.strip().decode('utf-8')) if self.verbose else None
        # wait for the subprocess to exit
        process.wait()

        y_pred = np.loadtxt(output_results, dtype=float)
        shutil.rmtree(tmp_dir)

        return y_pred

    def get_params(self, deep=True):
        params = dict(self.param)
        params['exec_path'] = self.exec_path
        params['config'] = self.config
        params['model'] = self.model
        params['verbose'] = self.verbose
        if 'output_model' in params:
            del params['output_model']
        return params

    def set_params(self, **kwargs):
        params = self.get_params()
        params.update(kwargs)
        self.__init__(**params)
        return self

    def feature_importance(self, feature_names=[], importance_type='weight'):
        """Get feature importance of each feature.
        Importance type can be defined as:
            'weight' - the number of times a feature is used to split the data across all trees.
            'gain' - the average gain of the feature when it is used in trees
            'cover' - the average coverage of the feature when it is used in trees

        Parameters
        ----------
        feature_names: list (optional)
           List of feature names.
        importance_type: string
            The type of feature importance
        """
        assert importance_type in [
            'weight'], 'For now, only weighted feature importance is implemented'

        match = re.findall("Column_(\d+)=(\d+)", self.model)

        if importance_type == 'weight':
            if len(match) > 0:
                dic_fi = {int(k): int(value) for k, value in match}
                if len(feature_names) > 0:
                    dic_fi = {feature_names[key]: dic_fi[key] for key in dic_fi}
            else:
                dic_fi = {}

        return dic_fi


class GBMClassifier(GenericGBM, ClassifierMixin):
    def __init__(self, exec_path="lightgbm",
                 config="",
                 application='binary',
                 n_estimators=10,
                 learning_rate=0.1,
                 num_leaves=127,
                 tree_learner="serial",
                 num_threads=1,
                 min_data_in_leaf=100,
                 metric='binary_logloss,',
                 is_training_metric='False',
                 feature_fraction=1.,
                 feature_fraction_seed=2,
                 bagging_fraction=1.,
                 bagging_freq=0,
                 bagging_seed=3,
                 metric_freq=1,
                 early_stopping_round=0,
                 max_bin=255,
                 is_unbalance=False,
                 num_class=1,
                 boosting_type='gbdt',
                 min_sum_hessian_in_leaf=10,
                 drop_rate=0.01,
                 drop_seed=4,
                 max_depth=-1,
                 lambda_l1=0.,
                 lambda_l2=0.,
                 min_gain_to_split=0.,
                 verbose=True,
                 model=None,
                 random_state=1,
                 train_file: str = None):
        super(GBMClassifier, self).__init__(exec_path=exec_path,
                                            config=config,
                                            application=application,
                                            n_estimators=n_estimators,
                                            learning_rate=learning_rate,
                                            num_leaves=num_leaves,
                                            tree_learner=tree_learner,
                                            num_threads=num_threads,
                                            min_data_in_leaf=min_data_in_leaf,
                                            metric=metric,
                                            is_training_metric=is_training_metric,
                                            feature_fraction=feature_fraction,
                                            feature_fraction_seed=feature_fraction_seed,
                                            bagging_fraction=bagging_fraction,
                                            bagging_freq=bagging_freq,
                                            bagging_seed=bagging_seed,
                                            metric_freq=metric_freq,
                                            early_stopping_round=early_stopping_round,
                                            max_bin=max_bin,
                                            is_unbalance=is_unbalance,
                                            num_class=num_class,
                                            boosting_type=boosting_type,
                                            min_sum_hessian_in_leaf=min_sum_hessian_in_leaf,
                                            drop_rate=drop_rate,
                                            drop_seed=drop_seed,
                                            max_depth=max_depth,
                                            lambda_l1=lambda_l1,
                                            lambda_l2=lambda_l2,
                                            min_gain_to_split=min_gain_to_split,
                                            verbose=verbose,
                                            model=model,
                                            random_state=random_state,
                                            train_file=train_file)

    def predict_proba(self, X):
        # 创建临时目录
        tmp_dir = tempfile.mkdtemp()
        # 数据是否稀疏
        issparse = sps.issparse(X)
        f_format = "svm" if issparse else "csv"

        predict_filepath = os.path.abspath(
            os.path.join(tmp_dir, "X_to_pred.{}".format(f_format)))
        output_model = os.path.abspath(os.path.join(tmp_dir, "model"))
        conf_filepath = os.path.join(tmp_dir, "predict.conf")
        output_results = os.path.abspath(
            os.path.join(tmp_dir, "LightGBM_predict_result.txt"))

        with open(output_model, mode="w") as file:
            file.write(self.model)

        io_utils.dump_data(X, np.zeros(X.shape[0]), predict_filepath, issparse)

        calls = [
            "task = predict\n",
            "data = {}\n".format(predict_filepath),
            "input_model = {}\n".format(output_model),
            "output_result={}\n".format(output_results)
        ]

        with open(conf_filepath, 'w') as f:
            f.writelines(calls)

        process = subprocess.Popen(
            [self.exec_path, "config={}".format(conf_filepath)],
            stdout=subprocess.PIPE, bufsize=1)

        with process.stdout:
            for line in iter(process.stdout.readline, b''):
                print(line.strip().decode('utf-8')) if self.verbose else None
        # wait for the subprocess to exit
        process.wait()

        raw_probabilities = np.loadtxt(output_results, dtype=float)

        if self.param['application'] == 'multiclass':
            y_prob = raw_probabilities

        elif self.param['application'] == 'binary':
            probability_of_one = raw_probabilities
            probability_of_zero = 1 - probability_of_one
            y_prob = np.transpose(
                np.vstack((probability_of_zero, probability_of_one)))
        else:
            raise ValueError('No label type has been given.')

        shutil.rmtree(tmp_dir)
        return y_prob

    def predict(self, X):
        # 使用predict_proba的概率最大列列号
        y_prob = self.predict_proba(X)
        return y_prob.argmax(-1)


class GBMRegressor(GenericGBM, RegressorMixin):
    def __init__(self, exec_path="lightgbm",
                 config="",
                 application='regression',
                 n_estimators=10,
                 learning_rate=0.1,
                 num_leaves=127,
                 tree_learner="serial",
                 num_threads=1,
                 min_data_in_leaf=100,
                 metric='l2,',
                 is_training_metric=False,
                 feature_fraction=1.,
                 feature_fraction_seed=2,
                 bagging_fraction=1.,
                 bagging_freq=0,
                 bagging_seed=3,
                 metric_freq=1,
                 early_stopping_round=0,
                 max_bin=255,
                 is_unbalance=False,
                 num_class=1,
                 boosting_type='gbdt',
                 min_sum_hessian_in_leaf=10,
                 drop_rate=0.01,
                 drop_seed=4,
                 max_depth=-1,
                 lambda_l1=0.,
                 lambda_l2=0.,
                 min_gain_to_split=0.,
                 verbose=True,
                 model=None,
                 random_state=1,
                 train_file: str = None):
        super(GBMRegressor, self).__init__(exec_path=exec_path,
                                           config=config,
                                           application=application,
                                           n_estimators=n_estimators,
                                           learning_rate=learning_rate,
                                           num_leaves=num_leaves,
                                           tree_learner=tree_learner,
                                           num_threads=num_threads,
                                           min_data_in_leaf=min_data_in_leaf,
                                           metric=metric,
                                           is_training_metric=is_training_metric,
                                           feature_fraction=feature_fraction,
                                           feature_fraction_seed=feature_fraction_seed,
                                           bagging_fraction=bagging_fraction,
                                           bagging_freq=bagging_freq,
                                           bagging_seed=bagging_seed,
                                           metric_freq=metric_freq,
                                           early_stopping_round=early_stopping_round,
                                           max_bin=max_bin,
                                           is_unbalance=is_unbalance,
                                           num_class=num_class,
                                           boosting_type=boosting_type,
                                           min_sum_hessian_in_leaf=min_sum_hessian_in_leaf,
                                           drop_rate=drop_rate,
                                           drop_seed=drop_seed,
                                           max_depth=max_depth,
                                           lambda_l1=lambda_l1,
                                           lambda_l2=lambda_l2,
                                           min_gain_to_split=min_gain_to_split,
                                           verbose=verbose,
                                           model=model,
                                           random_state=1,
                                           train_file=train_file)
