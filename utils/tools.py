import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil
import os
from tqdm import tqdm

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0, 
                save_mode=True, mode='min'):
        """
        Args:
            mode (str): 'min'或'max'，分别对应监控指标需要最小化或最大化
        """
        self.mode = mode
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.save_mode = save_mode
        
        # 根据模式初始化最佳值
        if self.mode == 'min':
            self.val_loss_min = np.Inf
        else:
            self.val_loss_min = -np.Inf

    def __call__(self, val_score, model, path):
        # 根据模式选择评分计算方式
        if self.mode == 'min':
            score = -val_score  # 越小越好
        else:
            score = val_score    # 越大越好

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            status_msg = f'EarlyStopping counter: {self.counter}/{self.patience}'
            
            if self.accelerator:
                self.accelerator.print(status_msg)
            else:
                print(status_msg)
                
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model, path)
            self.counter = 0

    def save_checkpoint(self, val_score, model, path):
        """保存模型检查点"""
        # 根据模式生成比较信息
        if self.mode == 'min':
            compare_sign = 'decreased' if val_score < self.val_loss_min else 'increased'
        else:
            compare_sign = 'increased' if val_score > self.val_loss_min else 'decreased'
        
        info_msg = (f'Validation score {compare_sign} '
                f'({self.val_loss_min:.6f} --> {val_score:.6f}). Saving model...')
        
        if self.verbose:
            if self.accelerator:
                self.accelerator.print(info_msg)
            else:
                print(info_msg)
        
        # 保存模型逻辑
        if self.accelerator:
            model = self.accelerator.unwrap_model(model)
            torch.save(model.state_dict(), f"{path}/checkpoint")
        else:
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(model.state_dict(), f"{path}/checkpoint")
        
        # 更新最佳值记录
        self.val_loss_min = val_score


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def del_files(dir_path):
    shutil.rmtree(dir_path)


def vali(args, model, vali_data, vali_loader, criterion, mae_metric):
    total_loss = []
    total_mae_loss = []
    model.eval()
    device = "cuda"
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                device)
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            # outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

            pred = outputs.detach()
            true = batch_y.detach()

            loss = criterion(pred, true)

            mae_loss = mae_metric(pred, true)

            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())

    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)

    model.train()
    return total_loss, total_mae_loss


def test(args, model, train_loader, vali_loader, criterion):
    x, _ = train_loader.dataset.last_insample_window()
    y = vali_loader.dataset.timeseries
    device = "cuda"
    x = torch.tensor(x, dtype=torch.float32).to(device)
    x = x.unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        B, _, C = x.shape
        dec_inp = torch.zeros((B, args.pred_len, C)).float().to(device)
        dec_inp = torch.cat([x[:, -args.label_len:, :], dec_inp], dim=1)
        outputs = torch.zeros((B, args.pred_len, C)).float().to(device)
        id_list = np.arange(0, B, args.eval_batch_size)
        id_list = np.append(id_list, B)
        for i in range(len(id_list) - 1):
            outputs[id_list[i]:id_list[i + 1], :, :] = model(
                x[id_list[i]:id_list[i + 1]],
                None,
                dec_inp[id_list[i]:id_list[i + 1]],
                None
            )
        # accelerator.wait_for_everyone()
        # outputs = accelerator.gather_for_metrics(outputs)
        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        pred = outputs
        true = torch.from_numpy(np.array(y)).to(device)
        batch_y_mark = torch.ones(true.shape).to(device)
        # true = accelerator.gather_for_metrics(true)
        # batch_y_mark = accelerator.gather_for_metrics(batch_y_mark)

        loss = criterion(x[:, :, 0], args.frequency_map, pred[:, :, 0], true, batch_y_mark)

    model.train()
    return loss


def load_content(args):
    if 'ETT' in args.data:
        file = 'ETT'
    else:
        file = args.data
    with open('./dataset/prompt_bank/{0}.txt'.format(file), 'r', encoding='utf-8') as f:
        content = f.read()
    return content