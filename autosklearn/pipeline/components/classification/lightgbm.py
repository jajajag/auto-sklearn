import numpy as np
import pandas as pd

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, \
    UniformFloatHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter
from ConfigSpace.conditions import EqualsCondition

from autosklearn.pipeline.components.base import (
    AutoSklearnClassificationAlgorithm,
    IterativeComponentWithSampleWeight,
)
from autosklearn.pipeline.implementations.models import GBMClassifier
from autosklearn.pipeline.constants import *
from sklearn.model_selection import train_test_split
import lightgbm as lgb


class LightGBM(
    IterativeComponentWithSampleWeight,
    AutoSklearnClassificationAlgorithm,
):
    def __init__(self,
                 # 1. 核心参数
                 boosting_type, n_estimators, learning_rate, num_leaves,
                 tree_learner, num_threads,
                 # 2. 学习控制参数
                 max_depth, min_data_in_leaf, min_sum_hessian_in_leaf,
                 bagging_freq, lambda_l1, lambda_l2, min_gain_to_split,
                 feature_fraction, feature_fraction_seed, early_stopping_round,
                 # 3. IO参数
                 max_bin, verbose=False,
                 # 5. 测量参数
                 metric="", is_training_metric=False, metric_freq=1,
                 random_state=None):

        # 1. 核心参数
        # 设置estmator初始为空
        self.estimator = None
        # 设置默认为二分类
        self.application = 'binary'
        # 模型种类
        self.boosting_type = boosting_type
        # 计算的轮数
        self.n_estimators = n_estimators
        # 学习率
        self.learning_rate = learning_rate
        # 每棵树最大结点个数
        self.num_leaves = num_leaves
        # 树的学习方式
        self.tree_learner = tree_learner
        # 使用的线程数，默认使用OpenMP的线程数
        self.num_threads = num_threads
        # 设置随机数种子
        if random_state is None:
            self.seed = 1
        else:
            self.seed = random_state.randint(1, 10000, size=1)[0]

        # 2. 学习控制参数
        # 树的最大深度
        self.max_depth = max_depth
        # L1和L2正则化系数
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        # 每个节点存储的数据个数
        self.min_data_in_leaf = min_data_in_leaf
        # 每个节点存储的hessian个数
        self.min_sum_hessian_in_leaf = min_sum_hessian_in_leaf
        # 默认不使用bagging
        self.bagging_freq = bagging_freq
        # 信息增益的阈值
        self.min_gain_to_split = min_gain_to_split
        # 选取feature的个数
        self.feature_fraction = feature_fraction
        # 选取特征用的随机数种子
        self.feature_fraction_seed = feature_fraction_seed
        # 默认不使用早停
        self.early_stopping_round = early_stopping_round

        # 3. IO参数
        # 默认不显示计算过程
        self.verbose = verbose
        # 最大bin的个数
        self.max_bin = max_bin

        # 4. 对象参数
        # 分类的总数(仅在muticlass时有用)
        self.num_class = 1
        # 二分类输出是否平衡
        self.is_unbalance = False

        # 5. 测量参数
        # 默认使用模型对应的metric
        self.metric = metric
        # 默认不显示训练集loss
        self.is_training_metric = is_training_metric
        # 输出metric的频率
        self.metric_freq = metric_freq

    #def iterative_fit(self, X, y, n_iter=2, refit=False, sample_weight=None):
    def fit(self, X, y):

        # 均衡的分离训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=1, test_size=0.1, stratify=y)
        # 将数据转化为lightgbm的格式
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test)
        refit = False
        self.estimator = None

        # 如果refit，则重新创建模型
        if refit:
            self.estimator = None

        # 如果没有创建模型，则新建模型
        if self.estimator is None:

            # 将所有参数格式化
            self.n_estimators = int(self.n_estimators)
            self.learning_rate = float(self.learning_rate)
            self.num_leaves = int(self.num_leaves)
            self.num_threads = int(self.num_threads)
            self.seed = int(self.seed)
            self.max_depth = int(self.max_depth)
            self.lambda_l1 = float(self.lambda_l1)
            self.lambda_l2 = float(self.lambda_l2)
            self.min_data_in_leaf = int(self.min_data_in_leaf)
            self.min_sum_hessian_in_leaf = float(self.min_sum_hessian_in_leaf)
            self.bagging_freq = int(self.bagging_freq)
            self.min_gain_to_split = float(self.min_gain_to_split)
            self.feature_fraction = float(self.feature_fraction)
            self.feature_fraction_seed = int(self.feature_fraction_seed)
            self.early_stopping_round = int(self.early_stopping_round)
            self.verbose = bool(self.verbose)
            self.max_bin = int(self.max_bin)
            self.is_training_metric = bool(self.is_training_metric)
            self.metric_freq = int(self.metric_freq)

            # 判断输出来选择是否用多分类
            if len(np.unique(y)) == 2:
                # 使用二分类
                self.application = 'binary'
                # 判断数据是否平衡，三倍以上则不平衡
                y_count = pd.value_counts(y)
                self.is_unbalance = (y_count[0] > 3 * y_count[1])

            else:
                # 使用多分类
                self.application = 'multiclass'
                # 计算分类数
                self.num_class = len(np.unique(y))

            # 超参数写入字典
            arguments = dict(
                # 1. 核心参数
                application=self.application,
                boosting_type=self.boosting_type,
                # 每次只跑两轮(别名n_estimators)
                #n_estimators=n_iter,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                tree_learner=self.tree_learner,
                num_threads=self.num_threads,
                random_state=self.seed,
                # 2. 学习控制参数
                max_depth=self.max_depth,
                lambda_l1=self.lambda_l1,
                lambda_l2=self.lambda_l2,
                min_data_in_leaf=self.min_data_in_leaf,
                min_sum_hessian_in_leaf=self.min_sum_hessian_in_leaf,
                bagging_freq=self.bagging_freq,
                min_gain_to_split=self.min_gain_to_split,
                feature_fraction=self.feature_fraction,
                feature_fraction_seed=self.feature_fraction_seed,
                early_stopping_round=self.early_stopping_round,
                # 3. IO参数
                verbose=1 if self.verbose else 0,
                max_bin=self.max_bin,
                # 4. 对象参数
                num_class=self.num_class,
                is_unbalance=self.is_unbalance,
                # 5. 测量参数
                metric=self.metric,
                is_training_metric=self.is_training_metric,
                metric_freq=self.metric_freq
            )
            # 创建分类器并调用
            # self.estimator = GBMClassifier(**arguments)
            # 轮数写死为100
            self.estimator = lgb.train(arguments, train_data,
                                       num_boost_round=100,
                                       valid_sets=test_data)

        """
        # 如果没有一次把所有轮数跑完
        elif not self.configuration_fully_fitted():
            self.n_estimators -= n_iter

            self.estimator.n_estimators += n_iter
            self.estimator.n_estimators = min(self.estimator.n_estimators,
                                              self.n_estimators)

        # 对模型进行fit(我怎么多写了一行。。)
        # self.estimator.fit(X_train, y_train, test_data=[(X_test, y_test)])

        # 不要忘了把flag设成true，不然会一直跑
        if self.estimator.n_estimators >= self.n_estimators:
            self.fully_fit_ = True
        """

        return self

    # 判断是否已经跑完所有轮数
    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        elif not hasattr(self, 'fully_fit_'):
            return False
        else:
            return self.fully_fit_

    # 返回label形式的预测值
    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        # return self.estimator.predict(X)
        proba = self.predict_proba(X)
        return np.argmin([proba, 1 - proba], axis=0)

    # 返回带概率的预测值
    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        # 调用上层GBMClassifier的predict_proba函数
        # return self.estimator.predict_proba(X)
        return self.estimator.predict(X)

    # 获得LightGBM的基本信息
    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'LightGBM',
                'name': 'Light Gradient Boosting Machine',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    # 获取超参数的搜索空间
    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):

        cs = ConfigurationSpace()

        # 1. 核心参数
        # iterations trees for multi-class classification problems
        n_estimators = UnParametrizedHyperparameter(name="n_estimators",
                                                    value=100)
        '''
        boosting_type = CategoricalHyperparameter(
            'boosting_type', ['gbdt', 'gbrt', 'rf', 'random_forest', 'dart',
                              'goss'], default_value='gbdt')
        '''
        # 只使用gbdt
        boosting_type = UnParametrizedHyperparameter(name="boosting_type",
                                                     value='gbdt')
        # Shrinkage rate
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=1e-6, upper=1, default_value=0.1,
            log=True)
        # Number of boosting iterations, LightGBM constructs num_class * num_
        # Max number of leaves in one tree
        num_leaves = UniformIntegerHyperparameter(
            name="num_leaves", lower=7, upper=8191, default_value=31)
        '''
        tree_learner = CategoricalHyperparameter(
            name='tree_learner', ['serial', 'feature', 'data', 'voting'],
            default_value='serial')
        '''
        # 只使用连续线性的learner
        tree_learner = UnParametrizedHyperparameter(name='tree_learner',
                                                    value='serial')
        # Using default number of threads in OpenMP
        num_threads = UnParametrizedHyperparameter(
            name="num_threads", value=0)

        # 2. 学习控制参数
        # limit the max depthfor tree model.This is used to deal with
        # over-fitting when  # data is small. Tree still grows leaf-wise
        # 默认不限制树的深度
        max_depth = UnParametrizedHyperparameter(name="max_depth", value=-1)
        # Minimal number of data in one leaf. Can be used to deal with
        # over-fitting
        min_data_in_leaf = UniformIntegerHyperparameter(
            name="min_data_in_leaf", lower=10, upper=100, default_value=20)
        # Minimal sum hessian in one leaf
        min_sum_hessian_in_leaf = UniformFloatHyperparameter(
            name="min_sum_hessian_in_leaf", lower=1e-5, upper=100,
            default_value=1e-3, log=True)
        # Frequency for bagging
        # 默认不使用bagging
        bagging_freq = UnParametrizedHyperparameter(name="bagging_freq",
                                                    value=0)
        # 写成超参格式，但不参与搜索
        # Like feature_fraction, but this will randomly select part of data
        # without resampling
        bagging_fraction = UniformFloatHyperparameter(
            name="bagging_fraction", lower=1e-10, upper=1, default_value=1,
            log=True)
        # Random seed for bagging
        bagging_seed = UnParametrizedHyperparameter(name="bagging_seed",
                                                    value=3)
        # L1 regularization
        lambda_l1 = UniformFloatHyperparameter(
            name="lambda_l1", lower=1e-10, upper=0.1, default_value=1e-10,
            log=True)
        # L2 regularization
        lambda_l2 = UniformFloatHyperparameter(
            name="lambda_l2", lower=1e-10, upper=0.1, default_value=1e-10,
            log=True)
        # The minimal gain to perform split
        min_gain_to_split = UniformFloatHyperparameter(
            name="min_gain_to_split", lower=1e-10, upper=1, default_value=1e-10,
            log=True)
        # 只在dart模式启用(默认不启用)
        # Dropout rate: a fraction of previous trees to drop during the dropout
        drop_rate = UniformFloatHyperparameter(
            name="drop_rate", lower=1e-10, upper=1, default_value=0.1,
            log=True)
        # Random seed to choose dropping models
        drop_seed = UnParametrizedHyperparameter(name="drop_seed", value=4)
        # LightGBM will randomly select part of features on each iteration if
        # feature_fraction smaller than 1.0. For example, if you set it to 0.8,
        # LightGBM will select 80% of features before training each tree
        feature_fraction = UniformFloatHyperparameter(
            name="feature_fraction", lower=0.1, upper=1, default_value=1,
            log=True)
        # Random seed for feature_fraction
        feature_fraction_seed = UnParametrizedHyperparameter(
            name="feature_fraction_seed", value=2)
        # Will stop training if one metric of one validation data doesn’t
        # improve in last early_stopping_round rounds
        # 默认不使用早停
        early_stopping_round = UnParametrizedHyperparameter(
            name="early_stopping_round", value=0)

        # 3. IO参数
        # Max number of bins that feature values will be bucketed in
        max_bin = UniformIntegerHyperparameter(
            name="max_bin", lower=7, upper=8191, default_value=255)

        cs.add_hyperparameters([
            # 1. 核心参数
            boosting_type, n_estimators, learning_rate, num_leaves,
            tree_learner, num_threads,
            # 2. 学习控制参数
            max_depth, min_data_in_leaf, min_sum_hessian_in_leaf,
            bagging_freq, lambda_l1, lambda_l2, min_gain_to_split,
            feature_fraction, feature_fraction_seed, early_stopping_round,
            # 3. IO参数
            max_bin
        ])

        '''
        # drop只和dart有关，加入限制条件
        drop_rate_condition = EqualsCondition(
            drop_rate, boosting_type, 'dart',
        )
        drop_seed_condition = EqualsCondition(
            drop_seed, boosting_type, 'dart',
        )
        cs.add_conditions([
            drop_rate_condition, drop_seed_condition
        ])
        '''

        return cs
