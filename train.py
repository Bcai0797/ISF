import torch
import numpy as np
import os
from dataseq import StrDataSequence, CSVDataSequence, NpyDataSequence
from model import ISF, ISF_concat
from utils import getCItd

torch.cuda.set_device(3)

MODEL_DIR = './models/'
MAX_STOP = 50
SAVE_STEP = 10000
summary_valid = {'CI':0, 'AUC':0, 'ANLP':float('inf')}
best_metrics = {'CI':0, 'AUC':0, 'ANLP':float('inf')}
epoch_metrics = {'CI':0, 'AUC':0, 'ANLP':float('inf')}
MODE_PT = 'cdf'

def logs_loss(y_true, y_pred):
    s_t = torch.clip(torch.sum(y_true * y_pred, dim=-1), 1e-9, 1)
    log_s = -torch.log(s_t)
    return torch.mean(log_s)

def time_mean_var(y_pred, t_list):
    n_batch = y_pred.size(0)
    t_list = t_list.to(y_pred.device).expand(n_batch, -1)
    mean = torch.sum(t_list * y_pred, dim=-1, keepdim=True)
    var = torch.sum(torch.pow(t_list - mean, 2) * y_pred, dim=-1)
    return mean, var


def train(save_dir, model, train_loader, vali_loader, optimizer, lr_scheduler, count_step, count_stop, n_batch, t_min, t_max, t_step=1, flag_hot=False):
    model.train()
    t_list = torch.arange(t_min, t_max-t_step, t_step)
    count_sample = 0
    tot_loss = 0
    for i_sample, sample in enumerate(train_loader):
        x, y, ot, st, yt = sample
        x, y = x.cuda(), y.cuda()
        n_b = x.size(0)
        count_step += n_b
        count_sample += n_b
        if i_sample == 0:
            n_acc = (n_batch + n_b - 1) // n_b
            n_acc = max(n_acc, 1)

        p = model.p_simpson(x, t_min, t_max, t_step)
        loss = logs_loss(y, p)
        tot_loss += loss
        loss.backward()
        
        if i_sample % n_acc == 0 or i_sample == len(train_loader)-1:
            optimizer.step()
            optimizer.zero_grad()
            print("%d/%d: loss %.5f" % (i_sample//n_acc+1, (len(train_loader) + n_acc - 1) // n_acc, tot_loss / (i_sample+1)), flush=True, end='\r')
        if SAVE_STEP < len(train_loader) * n_b and count_step >= SAVE_STEP:
            count_step -= SAVE_STEP
            print()
            test(model, vali_loader, 0, t_max, t_step=1, flag_hot=flag_hot)
            count_stop = save_model(save_dir, model, optimizer, count_stop)
            if count_stop <= 0:
                break
    
    if lr_scheduler is not None:
        lr_scheduler.step()
    print()
    if SAVE_STEP >= len(train_loader):
        test(model, vali_loader, 0, t_max, t_step=1, flag_hot=flag_hot)
        count_stop = save_model(save_dir, model, optimizer, count_stop)
    return count_step, count_stop

def test(model, vali_loader, t_min, t_max, t_step=1, flag_hot=False):
    model.eval()
    p_list = []
    y_list = []
    ot_list = []
    st_list = []
    mean_list = []
    var_list = []
    
    cnt = 0
    with torch.no_grad():
        t_list = torch.arange(t_min, t_max-t_step, t_step)
        for i_sample, sample in enumerate(vali_loader):
            x, y, ot, st, yt = sample
            n_b = x.shape[0]
            cnt += n_b
            x, y, yt = x.cuda(), y.cuda(), yt.cuda()
            p = model.p_simpson(x, t_min, t_max, t_step)
            mean, var = time_mean_var(p, t_list)

            p, y = p.cpu().numpy(), y.cpu().numpy()
            mean, var = mean.cpu().numpy(), var.cpu().numpy()

            p_list.extend(p)
            y_list.extend(y)
            ot_list.extend(ot)
            st_list.extend(st)
            var_list.extend(var)
            mean_list.extend(mean)

            print("%d/%d" % (i_sample+1, len(vali_loader)), end='\r', flush=True)
            if len(vali_loader) * n_b > SAVE_STEP and cnt >= 2048:
                break
    CI = getCItd(p_list, ot_list, st_list, t_step, flag_hot)
    print()
    print('CI: %.5f, Mean: %.5f, Std: %.5f' % (CI, np.mean(mean_list), np.mean(np.sqrt(var_list))), flush=True)
    summary_valid['CI'] = CI

def save_model(save_dir, model, optimizer, count_stop):
    metrics_list = ['CI']
    compare_list = [1]
    count_stop -= 1

    for m_name, com_factor in zip(metrics_list, compare_list):
        best_value = best_metrics[m_name] 
        epoch_value = epoch_metrics[m_name]
        now_value = summary_valid[m_name]

        if now_value * com_factor > epoch_value * com_factor:
            count_stop = MAX_STOP
            epoch_metrics[m_name] = now_value

        if now_value * com_factor > best_value * com_factor:
            count_stop = MAX_STOP
            best_metrics[m_name] = now_value
            torch.save({'optimizer': optimizer.state_dict(),
                        'state_dict': model.state_dict()},
                        os.path.join(save_dir, m_name + '.ckpt'))
    
    return count_stop


if __name__ == '__main__':
    dataset_root = './data/'
    ename = 'isf'
    embed_dim = 32
    
    dataset_list = ['CLINIC']
    # dataset_list = ['MUSIC']
    # dataset_list = ['METABRIC%d' % i for i in range(5)]
    max_time_list = [400] 
    n_embed_list = [200000] * len(dataset_list)
    lr_list = [1e-3, 1e-4, 1e-5]
    w_decay_list = [1e-3, 1e-4, 1e-5]
    n_epoch_list = [10000]
    it_batch = 32
    batch_size_list = [8, 16, 32, 64, 128, 256]
    t_min = 0
    t_margin = 0
    t_step = 1
    flag_concat = False # True: concat time and sample feature; False: Positional Encoding of time + sample feature

    train_idx = 0

    for i, dataset_name in enumerate(dataset_list):
        max_time = max_time_list[i]
        n_embed = n_embed_list[i]

        time_list = np.arange(t_min, max_time, t_step)

        if dataset_name in ['CLINIC', 'MUSIC']:
            train_fn = dataset_root + dataset_name + '/train.yzbx.txt'
            vali_fn = dataset_root + dataset_name + '/test.yzbx.txt'
            train_seq = StrDataSequence(train_fn, time_list, time_margin=t_margin)
            vali_seq = StrDataSequence(vali_fn, time_list)
        elif 'METABRIC' in dataset_name:
            train_file = dataset_root + dataset_name + '/train.npy'
            vali_file = dataset_root + dataset_name + '/vali.npy'
            train_seq = CSVDataSequence(train_file, time_list, time_margin=t_margin)
            vali_seq = CSVDataSequence(vali_file, time_list)

        flag_embed = dataset_name in ['CLINIC', 'MUSIC']
        n_feat = train_seq.get_map_size()

        model_name = dataset_name + '_ISF_inr'
        save_dir = os.path.join(MODEL_DIR, ename + model_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
                    
        summary_valid.update({'CI':0, 'AUC':0, 'ANLP':float('inf')})
        best_metrics.update({'CI':0, 'AUC':0, 'ANLP':float('inf')})

        for lr in lr_list:
            for w_decay in w_decay_list:
                for n_epoch in n_epoch_list:
                    for batch_size in batch_size_list:
                        print('Train Idx: %d, lr: %f, w_decay: %f, batch_size: %d' % (train_idx, lr, w_decay, batch_size))
                        train_idx += 1
                        
                        epoch_metrics.update({'CI':0, 'AUC':0, 'ANLP':float('inf')})
                        count_stop = MAX_STOP
                        count_step = 0
                        tmp_it_bath = batch_size
                        train_loader = torch.utils.data.DataLoader(train_seq, batch_size=tmp_it_bath, shuffle=True, drop_last=True)
                        vali_loader = torch.utils.data.DataLoader(vali_seq, batch_size=tmp_it_bath, shuffle=False, drop_last=False)

                        if flag_concat:
                            model = ISF_concat(n_feat, 128, n_embed, embed_dim, t_min, max_time, flag_embed=flag_embed, flag_bn=True, dropout_p=0.5)
                        else:
                            model = ISF(n_feat, 128, n_embed, embed_dim, t_min, max_time, flag_embed=flag_embed, flag_bn=True, dropout_p=0.5)
                        model.cuda()
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
                        lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)

                        for i_epoch in range(n_epoch):
                            count_step, count_stop = train(save_dir, model, train_loader, vali_loader, optimizer, None, count_step, count_stop, batch_size, 0, max_time, t_step=t_step, flag_hot='METABRIC' in dataset_name)
                            if count_stop <= 0:
                                break
                        
                        optimizer.zero_grad()
                        torch.cuda.empty_cache()
                        del model, optimizer
                        del train_loader, vali_loader

        del train_seq, vali_seq

    print('End', flush=True)
