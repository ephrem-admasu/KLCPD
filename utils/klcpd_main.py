import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.utils.extmath import svd_flip, randomized_svd
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from .data import HankelDataset
import time
import os

# load env variables
from dotenv import load_dotenv
load_dotenv()

SAVE_MODEL_DIR_PATH = os.environ['SAVE_MODEL_DIR_PATH']
PREDS_DIR_PATH = os.environ['PREDS_DIR_PATH']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on {device.upper()}')


def median_heuristic(X, beta=0.5):
    max_n = min(30000, X.shape[0])
    D2 = euclidean_distances(X[:max_n], squared=True)
    med_sqdist = np.median(D2[np.triu_indices_from(D2, k=1)])
    beta_list = [beta**2, beta**1, 1, (1.0/beta)**1, (1.0/beta)**2]
    return [med_sqdist * b for b in beta_list]


class NetG(nn.Module):
    def __init__(self, var_dim, RNN_hid_dim, num_layers: int = 1):
        super().__init__()
        self.var_dim = var_dim
        self.RNN_hid_dim = RNN_hid_dim

        self.rnn_enc_layer = nn.GRU(
            self.var_dim, self.RNN_hid_dim, num_layers=num_layers, batch_first=True)
        self.rnn_dec_layer = nn.GRU(
            self.var_dim, self.RNN_hid_dim, num_layers=num_layers, batch_first=True)
        self.fc_layer = nn.Linear(self.RNN_hid_dim, self.var_dim)

    # X_p:   batch_size x wnd_dim x var_dim (Encoder input)
    # X_f:   batch_size x wnd_dim x var_dim (Decoder input)
    # h_t:   1 x batch_size x RNN_hid_dim
    # noise: 1 x batch_size x RNN_hid_dim
    def forward(self, X_p, X_f, noise):
        X_p_enc, h_t = self.rnn_enc_layer(X_p)
        X_f_shft = self.shft_right_one(X_f)
        hidden = h_t + noise
        Y_f, _ = self.rnn_dec_layer(X_f_shft, hidden)
        output = self.fc_layer(Y_f)
        return output

    def shft_right_one(self, X):
        X_shft = X.clone()
        X_shft[:, 0, :].data.fill_(0)
        X_shft[:, 1:, :] = X[:, :-1, :]
        return X_shft


class NetD(nn.Module):
    def __init__(self, var_dim, RNN_hid_dim, num_layers: int = 1):
        super(NetD, self).__init__()

        self.var_dim = var_dim
        self.RNN_hid_dim = RNN_hid_dim

        self.rnn_enc_layer = nn.GRU(
            self.var_dim, self.RNN_hid_dim, num_layers=num_layers, batch_first=True)
        self.rnn_dec_layer = nn.GRU(
            self.RNN_hid_dim, self.var_dim, num_layers=num_layers, batch_first=True)

    def forward(self, X):
        X_enc, _ = self.rnn_enc_layer(X)
        X_dec, _ = self.rnn_dec_layer(X_enc)
        return X_enc, X_dec


class KL_CPD(nn.Module):
    def __init__(self, D: int, critic_iters: int = 5,
                 lambda_ae: float = 1e-5, lambda_real: float = 1e-3,
                 p_wnd_dim: int = 3, f_wnd_dim: int = 2, sub_dim: int = 1, RNN_hid_dim: int = 15):
        super().__init__()
        self.p_wnd_dim = p_wnd_dim
        self.f_wnd_dim = f_wnd_dim
        self.sub_dim = sub_dim
        self.D = D
        self.var_dim = D * sub_dim
        self.critic_iters = critic_iters
        self.lambda_ae, self.lambda_real = lambda_ae, lambda_real
        self.RNN_hid_dim = RNN_hid_dim
        self.netD = NetD(self.var_dim, RNN_hid_dim)
        self.netG = NetG(self.var_dim, RNN_hid_dim)
        self.loss_g_list = []
        self.loss_d_list = []

    @property
    def device(self):
        return next(self.parameters()).device

    def __mmd2_loss(self, X_p_enc, X_f_enc):
        sigma_var = self.sigma_var

        # some constants
        n_basis = 1024
        gumbel_lmd = 1e+6
        cnst = math.sqrt(1. / n_basis)
        n_mixtures = sigma_var.size(0)
        n_samples = n_basis * n_mixtures
        batch_size, seq_len, nz = X_p_enc.size()

        # gumbel trick to get masking matrix to uniformly sample sigma
        # input: (batch_size*n_samples, nz)
        # output: (batch_size, n_samples, nz)
        def sample_gmm(W, batch_size):
            U = torch.FloatTensor(batch_size*n_samples,
                                  n_mixtures).uniform_().to(self.device)
            sigma_samples = F.softmax(U * gumbel_lmd, dim=1).matmul(sigma_var)
            W_gmm = W.mul(1. / sigma_samples.unsqueeze(1))
            W_gmm = W_gmm.view(batch_size, n_samples, nz)
            return W_gmm

        W = Variable(torch.FloatTensor(batch_size*n_samples,
                     nz).normal_(0, 1).to(self.device))
        # batch_size x n_samples x nz
        W_gmm = sample_gmm(W, batch_size)
        # batch_size x nz x n_samples
        W_gmm = torch.transpose(W_gmm, 1, 2).contiguous()
        # batch_size x seq_len x n_samples
        XW_p = torch.bmm(X_p_enc, W_gmm)
        # batch_size x seq_len x n_samples
        XW_f = torch.bmm(X_f_enc, W_gmm)
        z_XW_p = cnst * torch.cat((torch.cos(XW_p), torch.sin(XW_p)), 2)
        z_XW_f = cnst * torch.cat((torch.cos(XW_f), torch.sin(XW_f)), 2)
        batch_mmd2_rff = torch.sum((z_XW_p.mean(1) - z_XW_f.mean(1))**2, 1)
        return batch_mmd2_rff

    def forward(self, X_p: torch.Tensor, X_f: torch.Tensor):
        batch_size = X_p.size(0)

        X_p_enc, _ = self.netD(X_p)
        X_f_enc, _ = self.netD(X_f)
        Y_pred_batch = self.__mmd2_loss(X_p_enc, X_f_enc)

        return Y_pred_batch

    def predict(self, ts):
        dataset = HankelDataset(
            ts, self.p_wnd_dim, self.f_wnd_dim, self.sub_dim)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                X_p, X_f = [batch[key].float().to(self.device)
                            for key in ['X_p', 'X_f']]
                pred_val = self.forward(X_p, X_f).cpu().detach().numpy()
                preds.append(pred_val)
        return np.concatenate(preds)

    def fit(self, ts, start_epoch, svd_method, components, epoches: int = 100, lr: float = 1e-2, weight_clip: float = .1, weight_decay: float = 0., momentum: float = 0., dataset_name=None):
        print('***** Training *****')
        # must be defined in fit() method
        optG_adam = torch.optim.AdamW(
            self.netG.parameters(), lr=lr, weight_decay=weight_decay)
        # lr_scheduler_g = lr_scheduler.CosineAnnealingLR(optG, T_max=epoches, eta_min=3e-5)
        optD_adam = torch.optim.AdamW(
            self.netD.parameters(), lr=lr, weight_decay=weight_decay)
        # lr_scheduler_d = lr_scheduler.CosineAnnealingLR(optD, T_max=epoches, eta_min=3e-5)
        optD_rmsprop = torch.optim.RMSprop(
            self.netD.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        optG_rmsprop = torch.optim.RMSprop(
            self.netG.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        

        dataset = HankelDataset(
            ts, self.p_wnd_dim, self.f_wnd_dim, self.sub_dim)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        sigma_list = median_heuristic(dataset.Y_hankel, beta=.5)
        self.sigma_var = torch.FloatTensor(sigma_list).to(self.device)

        # tbar = trange(epoches)
        for epoch in tqdm(range(start_epoch, epoches)):
            for batch in dataloader:
                # Fit critic
                for p in self.netD.parameters():
                    p.requires_grad = True
                for p in self.netD.rnn_enc_layer.parameters():
                    p.data.clamp_(-weight_clip, weight_clip)
                # (D_mmd2_mean, mmd2_real_mean, real_L2_loss, fake_L2_loss) = self._optimizeD(batch, optD)
                # train after every 15 epochs
                # if epoch % 15 == 0:
                self._optimizeD(batch, optD_rmsprop)
                # G_mmd2_mean = 0
                # if np.random.choice(np.arange(self.critic_iters)) == 0:
                # Fit generator
                for p in self.netD.parameters():
                    p.requires_grad = False  # to avoid computation
                # G_mmd2_mean = self._optimizeG(batch, optG)
                self._optimizeG(batch, optG_rmsprop)
            
            optD_adam.step()
            optG_adam.step()          

            # saving model dict to file after every 5 epochs
            if dataset_name:
                if epoch % 2 == 0:
                    torch.save(self.netD.state_dict(
                    ), f'{SAVE_MODEL_DIR_PATH}/{dataset_name}/{svd_method}_{components}/netd_{epoch}.pt')
                    torch.save(self.netG.state_dict(
                    ), f'{SAVE_MODEL_DIR_PATH}/{dataset_name}/{svd_method}_{components}/netg_{epoch}.pt')
        print('***** Plotting losses for Generator and Discriminator models *****')
        self.plot_losses(reduction_method=svd_method,
                         components=components, dataset_name=dataset_name)
        # print('[%5d/%5d] D_mmd2 %.4e G_mmd2 %.4e mmd2_real %.4e real_L2 %.6f fake_L2 %.6f'
        #   % (epoch+1, epoches, D_mmd2_mean, G_mmd2_mean, mmd2_real_mean, real_L2_loss, fake_L2_loss))

    def _optimizeG(self, batch, opt, lr_scheduler=None, grad_clip: int = 5):
        X_p, X_f = [batch[key].float().to(self.device)
                    for key in ['X_p', 'X_f']]
        batch_size = X_p.size(0)

        # real data
        X_f_enc, X_f_dec = self.netD(X_f)

        # fake data
        noise = torch.FloatTensor(1, batch_size, self.RNN_hid_dim).uniform_(-1, 1).to(self.device)
        # noise = torch.FloatTensor(1, batch_size, self.RNN_hid_dim).normal_(0, 1).to(self.device)
        noise = Variable(noise)
        Y_f = self.netG(X_p, X_f, noise)
        Y_f_enc, Y_f_dec = self.netD(Y_f)

        # batchwise MMD2 loss between X_f and Y_f
        G_mmd2 = self.__mmd2_loss(X_f_enc, Y_f_enc)

        # update netG
        self.netG.zero_grad()
        lossG = G_mmd2.mean()
        # lossG = 0.0 * G_mmd2.mean()
        self.loss_g_list.append(lossG.data.item())
        lossG.backward()

        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), grad_clip)

        opt.step()
        if lr_scheduler:
            lr_scheduler.step()

        # return G_mmd2.mean().data.item()

    def _optimizeD(self, batch, opt, lr_scheduler=None, grad_clip: int = 5):
        X_p, X_f, Y_true = [batch[key].float().to(self.device)
                            for key in ['X_p', 'X_f', 'Y']]
        batch_size = X_p.size(0)

        # real data
        X_p_enc, X_p_dec = self.netD(X_p)
        X_f_enc, X_f_dec = self.netD(X_f)

        # fake data
        noise = torch.FloatTensor(1, batch_size, self.netG.RNN_hid_dim).uniform_(-1, 1).to(self.device)
        # noise = torch.FloatTensor(1, batch_size, self.netG.RNN_hid_dim).normal_(0, 1).to(self.device)
        noise = Variable(noise)  # total freeze netG
        torch.no_grad()
        Y_f = Variable(self.netG(X_p, X_f, noise).data)
        Y_f_enc, Y_f_dec = self.netD(Y_f)

        # batchwise MMD2 loss between X_f and Y_f
        D_mmd2 = self.__mmd2_loss(X_f_enc, Y_f_enc)

        # batchwise MMD loss between X_p and X_f
        mmd2_real = self.__mmd2_loss(X_p_enc, X_f_enc)

        # reconstruction loss
        real_L2_loss = torch.mean((X_f - X_f_dec)**2)
        fake_L2_loss = torch.mean((Y_f - Y_f_dec)**2)

        # update netD
        self.netD.zero_grad()
        lossD = D_mmd2.mean() - self.lambda_ae * (real_L2_loss + fake_L2_loss) - \
            self.lambda_real * mmd2_real.mean()
        lossD = -lossD
        self.loss_d_list.append(lossD.data.item())
        lossD.backward()

        torch.nn.utils.clip_grad_norm_(self.netD.parameters(), grad_clip)

        opt.step()
        if lr_scheduler:
            lr_scheduler.step()

        # return D_mmd2.mean().data.item(), mmd2_real.mean().data.item(), real_L2_loss.data.item(), fake_L2_loss.data.item()

    def plot_losses(self, reduction_method: str, components: int, dataset_name: str):
        curr_time = time.strftime("%Y_%m_%d_%H_%M_%S")

        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_g_list, label='loss_g')
        plt.plot(self.loss_d_list, label='loss_d')
        plt.legend()
        plt.savefig(
            f'{PREDS_DIR_PATH}/{curr_time}_{reduction_method}_{components}_{dataset_name}_loss.png')
        plt.show()


def svd_wrapper(Y, k, method='svds'):
    if method == 'svds':
        Ut, St, Vt = svds(Y, k)
        idx = np.argsort(St)[::-1]
        St = St[idx]  # have issue with sorting zero singular values
        Ut, Vt = svd_flip(Ut[:, idx], Vt[idx])
    elif method == 'random':
        Ut, St, Vt = randomized_svd(Y, k, random_state=0)
    else:
        Ut, St, Vt = np.linalg.svd(Y, full_matrices=False)
        # now truncate it to k
        Ut = Ut[:, :k]
        St = np.diag(St[:k])
        Vt = Vt[:k, :]

    return Ut, St, Vt


def get_reduced_data(dataset, components, svd_method):
    print(f'***** Original dataset shape: {dataset.shape} *****')
    X, _, _ = svd_wrapper(dataset, components, method=svd_method)
    print(f'***** Reduced dataset shape: {X.shape} *****')
    return X


def train_and_pred_dataset(dataset, dataset_name, svd_method, components, preload_model=False):
    '''If dataset name is not None, then save the model dict to file after each epoch. If preload_model is True, then load the model from file and continue training'''
    dimension = dataset.shape[1]
    start_epoch = 0
    model = KL_CPD(dimension).to(device)

    # get model state dict from the file
    if dataset_name:
        # check if folder exists if not then create it
        model_folder_path = f'{SAVE_MODEL_DIR_PATH}/{dataset_name}/{svd_method}_{components}/'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        dir_files = os.listdir(model_folder_path)
        print('-----------------', len(dir_files))
        # check if folder is empty and check if reset_model_folder is False
        if len(dir_files) > 0 and preload_model:
            # sort files that starts with netd and netg
            netd_files = [
                file for file in dir_files if file.startswith('netd')]
            netg_files = [
                file for file in dir_files if file.startswith('netg')]

            # sort files with latest timestamp
            netd_files = sorted(netd_files, key=lambda x: os.path.getmtime(
                model_folder_path + x), reverse=True)
            netg_files = sorted(netg_files, key=lambda x: os.path.getmtime(
                model_folder_path + x), reverse=True)

            # load the latest file from netd and netg
            net_d_params = torch.load(model_folder_path + netd_files[0])
            net_g_params = torch.load(model_folder_path + netg_files[0])

            # load params to model
            model.netD.load_state_dict(net_d_params)
            model.netG.load_state_dict(net_g_params)

            # get the epoch number from the model file name. if epoch number is different then choose the min epoch number
            start_epoch = min(int(netd_files[0].split(
                '_')[-1].split('.')[0]), int(netg_files[0].split('_')[-1].split('.')[0]))
            print(
                f'***** Loaded model from file: {svd_method}_{components}/{netd_files[0]} and {svd_method}_{components}/{netg_files[0]} with epoch {start_epoch} *****')

    model.fit(dataset, dataset_name=dataset_name, start_epoch=start_epoch,
              svd_method=svd_method, components=components)
    predictions = model.predict(dataset)
    return predictions


def save_preds(dataset, predictions, reduction_method, dataset_name, skip_components=0, save_preds=True):
    print('***** Saving Predictions *****')
    components = dataset.shape[1]
    # get the min and max values for y-axis
    min_y = float('inf')
    max_y = float('-inf')
    for i in range(components):
        if skip_components == i+1:
            continue
        min_y = min(min_y, min(dataset[:, i]))
        max_y = max(max_y, max(dataset[:, i]))

    for i in range(components):
        if skip_components == i+1:
            continue
        plt.subplot(components+1, 1, i+1)
        plt.plot(dataset[:, i])
        plt.title(f'Component {i+1}')
        plt.ylim([min_y-0.2, max_y+0.2])
        plt.subplot(components+1, 1, components+1)

    plt.plot(predictions)
    plt.title('MMD')
    plt.suptitle(
        f'{reduction_method} with {components-skip_components} component(s) visualization')
    plt.tight_layout()
    if save_preds:
        curr_time = time.strftime("%Y_%m_%d_%H_%M_%S")
        plt.savefig(
            f'{PREDS_DIR_PATH}/{curr_time}_{reduction_method}_{components-skip_components}_{dataset_name}.png')
    plt.show()
    print('***** DONE *****')
