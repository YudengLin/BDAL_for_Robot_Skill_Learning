from __future__ import print_function, division

import numpy as np
import time

from collections import Counter

from pybullet_tools.utils import elapsed_time
from learn_tools.active_learner import ActiveLearner, format_x, THRESHOLD, tail_confidence
from learn_tools.learner import rescale, DEFAULT_INTERVAL, TRANSFER_WEIGHT, NORMAL_WEIGHT
from learn_tools import helper
from learn_tools.uncertain_learner import *
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from bnnsrc.Stochastic_Gradient_Langevin_Dynamics.BDNN_model import *
# from bnnsrc.Stochastic_Gradient_Langevin_Dynamics.bbbmodel  import *
NEARBY_COEFFICIENT = 0.5
BETA = 0.5

CLASSIFIER = '_classifier'
REGRESSOR = '_regressor'
BNN = 'bnn'
RRAM_BNN = 'RRAM_bnn'

RF = 'rf'

BNN_CLASSIFIER = BNN + CLASSIFIER
RF_CLASSIFIER = RF + CLASSIFIER
RRAM_BNN_CLASSIFIER = RRAM_BNN + CLASSIFIER

CLASSIFIERS = (BNN_CLASSIFIER, RF_CLASSIFIER, RRAM_BNN_CLASSIFIER,)

BNN_REGRESSOR = BNN + REGRESSOR
RF_REGRESSOR = RF + REGRESSOR
RRAM_BNN_REGRESSOR = RRAM_BNN + REGRESSOR

REGRESSORS = (BNN_REGRESSOR, RF_REGRESSOR, RRAM_BNN_REGRESSOR,)

# TODO: dummy learners (e.g. returns the mean)
BNN_MODELS = (BNN_CLASSIFIER, BNN_REGRESSOR,RRAM_BNN_CLASSIFIER,RRAM_BNN_REGRESSOR,)
RF_MODELS = (RF_CLASSIFIER, RF_REGRESSOR,)

CLASSIFIER_INTERVAL = (0, +1)
CLASSIFIER_THRES = np.mean(CLASSIFIER_INTERVAL)

############################################################

def get_sklearn_model(model_type, hidden_layers=[50, 50], n_estimators=500, **kwargs):

    NTrainPoints = 60000
    use_cuda = torch.cuda.is_available()

    if model_type == RRAM_BNN_CLASSIFIER:
        Exp = {'SimulationMode': False,
               'offset_sample': 0,
               'level': None,
               'calc_quant_scale': False,
               'scale_factor': 1,
               'Relax_time': 0,
               'q': 0.125,  # float(sys.argv[2]),
               'op_cell': 1,  # int(sys.argv[3]),
               'scale': 1.1  # float(sys.argv[4])
               }
        # print(Exp)
        use_p = True
        prior_sig = 0.19
        print(prior_sig)
        lr = 1e-3
        return mBDNN_langevin_classifier(lr=lr, channels_in=1, side_in=11, cuda=use_cuda, classes=2,
                                            batch_size=512, prior_sig=prior_sig, N_train=NTrainPoints, nhid=50, use_p=use_p, Exp=Exp)

    raise ValueError(model_type)

############################################################

def retrain_sklearer(learner, newx=None, newy=None, new_w=None):
    start_time = time.time()
    if (newx is not None) and (newy is not None):
        learner.xx = np.vstack((learner.xx, newx))
        if learner.model_type in CLASSIFIERS:
            newy = (THRESHOLD < newy[0])
        learner.yy = np.hstack((learner.yy, newy))
        if new_w is not None:
            learner.weights = np.hstack((learner.weights, new_w))

    # TODO: what was this supposed to do?
    #self.xx, self.yy, sample_weight = shuffle(self.xx, self.yy, _get_sample_weight(self))
    if learner.model_type in CLASSIFIERS or learner.model_type in REGRESSORS:
    # if learner.model.__class__ in [MLPDropout_Classifier, MLPDropout_Regressor, BBB_classifier,BBB_Regressor, Net_RRAM_langevin_classifier]:
        learner.model.fit(learner.xx, learner.yy)
    else:
        # TODO: preprocess to change the mean_function
        # https://scikit-learn.org/stable/modules/preprocessing.html
        xx, yy = learner.xx, learner.yy
        weights = None
        if (learner.transfer_weight is not None) and (2 <= len(Counter(learner.weights))):
            assert 0. <= learner.transfer_weight <= 1.
            #weights = learner.weights
            weights = np.ones(yy.shape)
            total_weight = sum(weights)
            normal_indices = np.argwhere(learner.weights==NORMAL_WEIGHT)
            weights[normal_indices] = total_weight*(1 - learner.transfer_weight) / len(normal_indices)
            transfer_indices = np.argwhere(learner.weights==TRANSFER_WEIGHT)
            weights[transfer_indices] = total_weight*learner.transfer_weight/len(transfer_indices)

            print('Transfer weight: {:.3f} | Normal: {} | Transfer: {} | Other: {}'.format(
                learner.transfer_weight, len(normal_indices), len(transfer_indices),
                len(weights) - len(normal_indices) - len(transfer_indices)))
        #weights = 1./len(yy) * np.ones(len(yy))
        #num_repeat = 0
        # if num_repeat != 0:
        #     xx = np.vstack([xx, xx[:num_repeat]])
        #     yy = np.vstack([yy, yy[:num_repeat]])
        #     weights = np.concatenate([
        #         1./len(yy) * np.ones(len(yy)),
        #         1./num_repeat * np.ones(num_repeat),
        #     ])
        learner.model.fit(xx, yy, sample_weight=weights)
    print('Trained in {:.3f} seconds'.format(elapsed_time(start_time)))
    learner.metric()
    learner.trained = True

############################################################
num_predicts = 0
def bnn_predict(model, X):
    global num_predicts
    num_predicts +=20
    # print("predict %d"%(num_predicts))
    num_predict = 20
    predictions = [model.predict(X) for _ in np.arange(num_predict)]
    #if type(model) not in [ExtraTreesRegressor, ExtraTreesClassifier]: TODO: python2 naming issue
    #    raise ValueError(model)
    #if isinstance(model, ExtraTreesClassifier):
    if 'classifier' in model.__class__.__name__.lower():
        # pred = out.data.max(dim=1, keepdim=False)[1]
        lower, upper = DEFAULT_INTERVAL
        predictions = [lower + (upper - lower) * pred for pred in predictions]
    mu = np.mean(predictions, axis=0)
    var = np.var(predictions, axis=0)
    if any(var)==0:
        print('Got zero in var!')
        var[var==0] = min(var[var.nonzero()])
    return mu, var

############################################################

# TODO: the keras models don't seem to work well when using multiple processors

class ActiveBNN(UncertainLearner):
    def __init__(self, func, initx=np.array([[]]), inity=np.array([]),
                 model_type=BNN_CLASSIFIER, epochs=1000,
                 validation_split=0.1, batch_size=100,
                 use_sample_weights=False, verbose=True, sample_strategy=UNIFORM, **kwargs):
        print('{} using {}'.format(self.__class__.__name__, model_type))
        if model_type in CLASSIFIERS:
            inity = (THRESHOLD < inity).astype(float)
        self.sample_strategy = sample_strategy
        assert sample_strategy in SAMPLE_STRATEGIES, 'Invalid sample strategy.'
        super(ActiveBNN, self).__init__(func, initx=initx,inity=inity,**kwargs)
        self.model_type = model_type
        self.model = get_sklearn_model(model_type)
        self.use_sample_weights = use_sample_weights
        self.verbose = verbose
        self.name = 'bnn_{}'.format(model_type)
        self.epochs = epochs
        self.validation_split = validation_split
        self.batch_size = batch_size

    ##############################

    def predict_x(self, X, **kwargs):
        mu, _ = self.predict(X, **kwargs)
        return [{'mean': float(mu[i])} for i in range(X.shape[0])]

    @property
    def ard_parameters(self):
        return None

    @property
    def lengthscale(self):
        return None

    def predictive_gradients(self, x):

        raise NotImplementedError()

    def metric(self):
        mu, var = self.predict(self.xx)

        if 'classifier' in self.name:
            pred = (THRESHOLD < mu)
            label = (THRESHOLD < self.yy)
            tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
            acc = float(tn + tp) / len(label)
            print('accuracy = {}, fpr = {}, fnr = {}'.format(
                acc, float(fp)/(tn + fp), float(fn)/(tp + fn)))
            return acc
        elif 'regressor' in self.name:
            mse = mean_squared_error(mu,self.yy)
            print('mse = {}'.format(mse))
            pred = (THRESHOLD < mu)
            label = (THRESHOLD < self.yy)
            tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
            acc = float(tn + tp) / len(label)
            print('accuracy = {}, fpr = {}, fnr = {}'.format(
                acc, float(fp) / (tn + fp), float(fn) / (tp + fn)))
            return acc
        else:
           print('error in metric')
           return None

    def predict(self, X, noise=True):
        mu, var = bnn_predict(self.model, X)
        return mu, var

    def retrain(self, **kwargs):
        return retrain_sklearer(self, **kwargs)
