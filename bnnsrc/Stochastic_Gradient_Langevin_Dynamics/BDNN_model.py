# Some of the code is referenced from below Github files:
# https://github.com/YudengLin/memristorBDNN, which includes `BDNN_Model.py` and `memristor_model.py`
import torch.optim
from bnnsrc.base_net import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from optimizers import *
import torch.nn.functional as F
from  memristor_model import *
from gaussian import gaussian
from sklearn.base import BaseEstimator
from bnnsrc.utils import *

CLASSIFIER = '_classifier'
REGRESSOR = '_regressor'
RRAM_BNN = 'RRAM_bnn'

RRAM_BNN_CLASSIFIER = RRAM_BNN + CLASSIFIER
CLASSIFIERS = (RRAM_BNN_CLASSIFIER,)

RRAM_BNN_REGRESSOR = RRAM_BNN + REGRESSOR
REGRESSORS = (RRAM_BNN_REGRESSOR,)

RRAM_BNN_MODELS = (RRAM_BNN_CLASSIFIER, RRAM_BNN_REGRESSOR,)
def cross_entropy_loss(outputs, target, sample, location = 0.0, scale = 1.0, reduction=True):
    mlpdw, err = 0., 0.
    for s in range(sample):
        output_s = outputs[:,s,:,:].squeeze(1)
        mlpdw_s = F.cross_entropy(output_s, target, reduction='sum')
        mlpdw += mlpdw_s
        pred = output_s.data.max(dim=1, keepdim=False)[1]
        err += pred.ne(target.data).sum()
    acc = 1. - float(err)/(outputs.shape[0]*sample)
    return mlpdw, acc

class mBDNN(nn.Module):
    def __init__(self, input_dim, output_dim, n_hid, no_sample, prior=0.1, Exp=None):
        super(mBDNN, self).__init__()

        self.n_hid = n_hid

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gaussian_var = prior
        self.Exp = Exp
        self.layers = nn.ModuleList()
        self.layers.append(
            BayesLinear_memristorLayer(1, input_dim, 50, prior=gaussian(0, self.gaussian_var), Exp=self.Exp,
                                       no_sample=no_sample))
        self.layers.append(
            BayesLinear_memristorLayer(2, 50, 50, prior=gaussian(0, self.gaussian_var), Exp=self.Exp,
                                       no_sample=no_sample))
        self.layers.append(
            BayesLinear_memristorLayer(3, 50, 2, prior=gaussian(0, self.gaussian_var), Exp=self.Exp,
                                       no_sample=no_sample))

        self.layer_num = len(self.layers)

        self.act = nn.Sigmoid()

    def forward(self, x, no_sample):
        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        # x: [Batch, no_sample, 1, features]
        x = x.unsqueeze(1).unsqueeze(1).expand(-1, no_sample, -1, -1)
        # -----------------
        KL_loss_total = 0.
        for id, layer in enumerate(self.layers):
            x, KL_loss, _, _  = layer(x,no_sample)
            # -----------------
            KL_loss_total += KL_loss
            if id!= self.layer_num-1:
                x = self.act(x)

        return x, KL_loss_total

    def clip_win(self):
        with torch.no_grad():
            for layer in self.layers:
                layer.clip_win()

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        super(MyDataset, self).__init__()
        assert x.shape[0] == y.shape[0] # assuming shape[0] = dataset size
        self.x = x
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class mBDNN_langevin_classifier(BaseNet, BaseEstimator):
    eps = 1e-6

    def __init__(self, lr=1e-3, channels_in=3, side_in=28, cuda=True, classes=2, N_train=512, prior_sig=0.1,
                 nhid=50, use_p=False,Exp=None, batch_size=512):
        super(mBDNN_langevin_classifier, self).__init__()
        self.lr = lr
        self.lr_schedule = None
        self.em_schedule = None
        self.cuda = cuda
        self.channels_in = channels_in
        self.prior_sig = prior_sig
        self.classes = classes
        self.N_train = N_train
        self.side_in = side_in
        self.nhid = nhid
        self.use_p = use_p
        self.Exp=Exp
        self.batch_size = batch_size
        self.epoch = 50
        self.weight_set_samples = []
        self.test = False
        self.init = True

        self.create_net(Exp=Exp)
        self.create_opt(Exp=Exp)

    def create_net(self,Exp=None):
        self.no_sample = 10
        self.model = mBDNN(input_dim=self.side_in, output_dim=self.classes,
                           n_hid=self.nhid, no_sample=self.no_sample, prior=self.prior_sig, Exp=Exp)
        self.loss_func = cross_entropy_loss
        if self.cuda:
            self.model.cuda()

    def create_opt(self,Exp=None):
        self.epoch = 120
        if self.use_p:
            self.optimizer = SGLD(model=self.model, params=self.model.parameters(), lr=0.51, norm_sigma=self.prior_sig,
                                  centered=True, addnoise=True, Exp=Exp)
            # We use lr_scheduler to adjust percentage of updating cells
            self.p_schedule = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.9, last_epoch=-1,
                                                              verbose=True)
        else:
            exit(-1)
    def normalise(self, x, test=False, quant_bit=8):
        if not test:
            self.x_ave = np.mean(x,axis=0)
            self.x_std = np.std(x,axis=0)
            if quant_bit is not None:
                self.x_range = 2* (2.5*self.x_std)
                self.x_scale = self.x_range/(2**quant_bit-1)

        x_norm = (x-self.x_ave)/self.x_std
        if quant_bit is not None:
            x_norm = (x_norm/self.x_scale).round()*self.x_scale

        return x_norm

    def fit(self, x, y):
        self.set_mode_train(True)

        x = self.normalise(x)
        x, y = to_variable(var=(x, y), cuda=self.cuda)
        y = y.long()

        traindata = MyDataset(x, y)
        trainloader = torch.utils.data.DataLoader(traindata, batch_size=self.batch_size, shuffle=True,
                                                  num_workers=0)
        nb_train = len(trainloader)

        loss_list, acc_list = [],[]
        for e in range(self.epoch):
            beta =1.
            # if e%120==0:
            #     trainloader = torch.utils.data.DataLoader(traindata, batch_size=self.batch_size, shuffle=True,
            #                                               num_workers=0)
            #     nb_train = len(trainloader)
            #     self.optimizer = pSGLD(model=self.model, params=self.model.parameters(), lr=0.8,
            #                            norm_sigma=self.prior_sig,
            #                            centered=True, addnoise=True, Exp=self.Exp)
            #     self.p_schedule = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=25, gamma=0.9,
            #                                                       last_epoch=-1,
            #                                                       verbose=True)
            for k,(x_b, y_b) in enumerate(trainloader):
                self.optimizer.zero_grad()
                self.model.clip_win()
                net_out, KL_loss = self.model(x_b,self.no_sample)
                fit_loss_total, acc = self.loss_func(net_out, y_b, self.no_sample)
                acc_list.append(acc)
                total_loss = (beta*KL_loss + fit_loss_total)/self.no_sample
                loss_list.append(total_loss.detach().cpu().numpy())
                total_loss.backward()
                self.optimizer.step()
            metrics = acc_list[-nb_train*5:]
            if (not self.init) and e>10 and (np.mean(metrics)>0.999 and np.std(metrics)<0.005):
                print(e)
                break
            self.p_schedule.step()
        if self.init:
            self.init = False

        return self

    def predict(self, x, no_sample=1):
        self.set_mode_train(False)
        x = self.normalise(x, test=True)

        x, = to_variable(var=(x,), cuda=self.cuda)
        self.model.clip_win()

        net_out, _ = self.model(x, no_sample)
        probs = F.softmax(net_out, dim=3).data.cpu().detach().numpy().squeeze()
        return probs[:,1:]

    def score(self, x, y, train=False):
        self.set_mode_train(False)
        x, y = to_variable(var=(x, y), cuda=self.cuda)
        y = y.long()
        self.model.clip_win()

        net_out, KL_loss = self.model(x,1)

        loss, acc = self.loss_func(net_out, y, 1)

        return acc
