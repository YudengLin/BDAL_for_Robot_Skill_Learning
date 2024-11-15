# Some of the code is referenced from below Github files:
# https://github.com/YudengLin/memristorBDNN, which includes `BDNN_Model.py` and `memristor_model.py`
from __future__ import division, print_function
import platform_4k
import torch
import torch.nn as nn
import numpy as np

class operation_paras():
    def __init__(self):
        self.set_p = 0.5
        self.reset_p = 0.5
BNN_Chip = None
POSITIVE = 1
NEGATIVE = -1
NULLOP = 0
class Platform_device(nn.Module):
    def __init__(self, layer_id, input_dim, output_dim, N=3,init_states=None, Exp=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.N = N
        self.layer_id = layer_id
        global BNN_Chip
        if BNN_Chip is None:
            print('Reinit Platform_4K')
            print(Exp)
            BNN_Chip = platform_4k.Platform_4k(Exp)
            # is_test=False: will connect and use MNIST Platform in running time
            BNN_Chip.platform_init(init_states, is_test=False)

        self.XBArray_ID = BNN_Chip.deploy_XBArray(layer_id, input_dim, output_dim, N)
        self.device_sample = nn.Parameter(torch.ones(input_dim, output_dim, N))
        self.set_acc_count = 0
        self.reset_acc_count = 0
        self.acc_th = 4


    def __del__(self):
        global BNN_Chip
        BNN_Chip = None

    def sample(self, no_sample):
        # each weight = sum read current of N memristor devices
        # here we sample each weight for no_sample times

        # read_Ndevice: [no_sample, input_dim, output_dim, N]
        read_Ndevice = BNN_Chip.sample_layer(self.XBArray_ID, no_sample)
        with torch.no_grad():
            self.device_sample.set_(torch.from_numpy(read_Ndevice.astype(np.float32)).cuda())
        # read_sum_N: [no_sample, input_dim, output_dim]
        read_sum_N = torch.sum(self.device_sample, dim=3)
        # read_sum_N = torch.from_numpy(read_sum_N).cuda()
        read_sum_N_mus = torch.mean(read_sum_N,dim=0)
        read_sum_N_std = torch.std(read_sum_N,dim=0)

        # This is return samples, mus and std
        # v=std^2, v(x1+x2+...)=v(x1)+v(x2)+...
        return read_sum_N,read_sum_N_mus,read_sum_N_std,None # [no_sample * input_dim * output_dim]

    def set(self, set_mask, layer_id, op_cell=1):
        # op_dir: direction of operation: 'positive' or 'negative' [input_dim * output_dim * N]
        # set_mask: operation mask [input_dim * output_dim * N]
        if self.set_acc_count == 0:
            self.set_mask = set_mask.cpu().numpy().astype(int)
            self.set_acc_count += 1
        elif self.set_acc_count <= self.acc_th:
            self.set_mask += set_mask.cpu().numpy().astype(int)
            self.set_acc_count += 1
        else:
            self.set_mask += set_mask.cpu().numpy().astype(int)
            set_mask = self.set_mask >= self.acc_th/2
            # 1. Random select cells to op
            ind = torch.randperm(self.N)
            set_mask[:,:,ind[op_cell:]] = False
            np.apply_along_axis(np.random.shuffle, axis=2, arr=set_mask)
            self.set_acc_count = 0
            BNN_Chip.update_layer(POSITIVE, layer_id, set_mask)
        return None

    def reset(self, reset_mask, layer_id, op_cell=1):
        if self.reset_acc_count == 0:
            self.reset_mask = reset_mask.cpu().numpy().astype(int)
            self.reset_acc_count += 1

        elif self.reset_acc_count <= self.acc_th:
            self.reset_mask += reset_mask.cpu().numpy().astype(int)
            self.reset_acc_count += 1
        else:
            self.reset_mask += reset_mask.cpu().numpy().astype(int)
            reset_mask = self.reset_mask >= self.acc_th / 2
            # 1. Random select cells to op
            ind = torch.randperm(self.N)
            reset_mask[:, :, ind[op_cell:]] = False
            np.apply_along_axis(np.random.shuffle, axis=2, arr=reset_mask)
            self.reset_acc_count = 0
            BNN_Chip.update_layer(NEGATIVE, layer_id, reset_mask)
        return None

class XBArray(nn.Module):
    def __init__(self, layer_id, input_dim, output_dim, N=3, init_states=[1.4],offset=0., scale=1.,no_sample=None, Exp=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_id = layer_id
        self.devices = []
        self.N, self.offset, self.scale, self.no_sample = N, offset, scale, no_sample
        # # SimulationMode: True-use memristor paras model ; False- use platform
        self.Simulation_Mode = Exp['SimulationMode']
        if Exp['SimulationMode']:
            print('Simulation Mode is not implemented.')
            exit(-1)

        elif not Exp['SimulationMode']:
            self.XBArray_plat = Platform_device(layer_id, input_dim, output_dim, N,init_states, Exp)
            self.set_fn = self.XBArray_plat.set
            self.reset_fn = self.XBArray_plat.reset

    def sample(self, no_sample=None):
        if no_sample == None:
            no_sample = self.no_sample
        # sample weight
        ave_fly_time = 0.
        if self.Simulation_Mode:
            raw_weight_sample, raw_weight_mus, raw_weight_stds = self.devices.sample(no_sample)

        elif not self.Simulation_Mode:
            raw_weight_sample, raw_weight_mus, raw_weight_stds, ave_fly_time = self.XBArray_plat.sample(no_sample)
        weight_sample = ((raw_weight_sample - self.offset)/self.scale)
        weight_mus = ((raw_weight_mus - self.offset)/self.scale)
        weight_stds = (raw_weight_stds / self.scale)

        # This is return weight samples, mus and std
        return weight_sample, weight_mus, weight_stds, ave_fly_time

    def clip_win(self):
        if self.Simulation_Mode:
            self.devices.clip_win()
        elif not self.Simulation_Mode:
            return

    def forward(self, x, no_sample=None):
        # sample gaussian noise for each weight and each bias
        weight_sample, self.weight_mus, self.weight_stds, ave_fly_time = self.sample(no_sample)
        output = torch.matmul(x, weight_sample)

        return output, self.weight_mus, self.weight_stds, ave_fly_time

    def program(self, set_mask, reset_mask, layer_id=0, no_pulse=1, no_verify=0):
        # update device current state
        # target: raw target value [input_dim*output_dim*N]
        # update_mask: update mask (calc based on SNR or dSNR...) [input_dim*output_dim*N]
        set_tuned_state = self.set_fn(set_mask, layer_id)
        reset_tuned_state = self.reset_fn(reset_mask, layer_id)

class BayesLinear_memristorLayer(nn.Module):

    def __init__(self,layer_id, input_dim, output_dim, Exp=None, prior=None, N=3, offset=6.66,  no_sample=5):
        super(BayesLinear_memristorLayer, self).__init__()
        input_dim = input_dim + 1 # for bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior = prior
        self.layer_id = layer_id
        self.N, self.offset, self.no_sample = N, offset, no_sample
        init_states = [1.2, 2.2, 3.2]
        self.Exp = Exp
        self.scale = Exp['scale']
        self.XBAccelerator= XBArray(layer_id, input_dim, output_dim, N, init_states, offset, self.scale, no_sample, Exp)

        # layer update  op paras
        self.threshold, self.no_pulse, = 0., 1
        # self.threshold, self.no_pulse, self.no_verify = Exp['Thre'][Exp['NoExp']], 1, 5
        # print("Threshold = "+str(self.threshold))
        self.update_step = 0
        # self.B_XBAccelerator = XBArray(1, output_dim, N, init_states, offset, scale,Exp)

        self.calc_quant_scale = Exp['calc_quant_scale']
        # history of max x
        self.x_max_pre = torch.zeros(1).cuda()
        self.quant_scale = 1.#= nn.Parameter(torch.ones([]), requires_grad=False)
        # self.quant_scale = 1.
        self.acc_scale = 1.
        # self.calc_quant_scale = nn.Parameter(torch.ones(1),requires_grad=False)
        if layer_id == 0 or layer_id == 2:
            self.quant_min, self.quant_max = -127, 127
        else:
            self.quant_min, self.quant_max = 0, 255
        self.cut_factor = 1

    def clip_win(self):
        self.XBAccelerator.clip_win()


    def quant(self, x, pre_scale_factor=1.):
        self.acc_scale = pre_scale_factor * self.quant_scale

        bias = 1.
        if self.XBAccelerator.training:
            # training mode
            return x, bias
        else:
            if self.calc_quant_scale :
                # calc scale mode
                x_max = torch.max(abs(x.flatten()))
                if self.x_max_pre<x_max:
                    self.x_max_pre = x_max
                    # x*s = xq  s=xq/x
                    self.quant_scale.set_(((self.cut_factor * self.quant_max) / self.x_max_pre).cuda())
                    print('Layer ID %d quant_scale= %f' % (self.layer_id, self.quant_scale.cpu().numpy()))
                return x, bias

            else:
                if self.quant_scale==1. and pre_scale_factor==1.:
                    return x, bias
                else:
                    # inference mode with quantized x
                    # we convert input to quantized x: xq=x*s
                    x = (x * self.quant_scale).round().clamp_(self.quant_min,self.quant_max)
                    bias = (self.acc_scale).round().clamp_(self.quant_min,self.quant_max)
                    return x, bias

    def forward(self, x, no_sample=None, pre_scale_factor=1.):
        # when not training, we get the scale param of quantization or x is quantized
        x, bias = self.quant(x, pre_scale_factor)
        # input x now include 1 for bias
        dims = [dim for dim in x.size()[0:-1]]
        dims.append(1)
        x = torch.cat((x, (torch.ones(dims)*bias).cuda()), len(dims)-1)

        output, weight_mus, weight_stds, ave_fly_time = self.XBAccelerator(x, no_sample)

        # computing the KL loss term
        prior_cov, varpost_cov = self.prior.sigma ** 2, weight_stds ** 2
        KL_loss = 0.5 * (torch.log(prior_cov / varpost_cov)).sum() - 0.5 * weight_stds.numel()
        KL_loss = KL_loss + 0.5 * (varpost_cov / prior_cov).sum()
        KL_loss = KL_loss + 0.5 * ((weight_mus - self.prior.mu) ** 2 / prior_cov).sum()

        return output, KL_loss, self.acc_scale, ave_fly_time

    def program_conductance(self, err_margin, delta_sample, target, no_verify, p_adjust, Exp=None):
        # no_verify = 0: program without verify
        i_set = torch.zeros_like(target)
        i_reset = torch.zeros_like(target)
        i_read_sum = torch.zeros_like(target)
        fail_mask = torch.ones_like(target)

        self.program_times = torch.zeros_like(target)

        q_th = Exp['q']*p_adjust
        op_cell = Exp['op_cell']
        if no_verify ==0:
            set_mask, reset_mask = self.abs_percentile_th(delta_sample, q_th, op_cell)
            self.XBAccelerator.program(set_mask, reset_mask, self.layer_id)
        else:
            print('write with verify is not implemented.')
            exit(-1)

        return  self.program_times, fail_mask, i_set,i_reset, i_read_sum

    def abs_percentile_th(self, x, q, op_cell):
        if not self.Exp['SimulationMode']:
            x = torch.mean(x, dim=0)
            xf = abs(x.flatten())
            # q_op = torch.quantile(xf, 1.-q, interpolation='midpoint')
            q_op = torch.quantile(xf, 1. - q, dim=None, keepdim=False, interpolation='midpoint')

            x_set = x > q_op
            x_reset = x < -q_op

        else:
            xf = abs(x.flatten())
            # q_op = torch.quantile(xf, 1.-q, interpolation='midpoint')
            q_op = torch.quantile(xf, 1.-q, dim=None,keepdim=False,interpolation='midpoint')

            x_set = x > q_op
            x_reset = x < -q_op

            # 1. Random select cells to op
            ind = torch.randperm(self.N)
            x_set[:,:,ind[op_cell:]] = False
            x_reset[:,:,ind[op_cell:]] = False

        # 2. First cell to Reset; last cell to Set
        # x_set[:,:,[0,1]] = False
        # x_reset[:,:,[1,2]] = False

        return x_set, x_reset



