import argparse
import torch
# from accelerate import Accelerator, DeepSpeedPlugin
# from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from utils.tools import visual
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from datetime import datetime
from models import Autoformer, DLinear, TimeLLM

from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content

parser = argparse.ArgumentParser(description='Time-LLM')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, DLinear]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='BERT', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='768', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768


# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)

args = parser.parse_args()
# ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
# deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
# accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)
device = "cuda"
if args.is_training:
    for ii in range(args.itr):
    # setting record of experiments
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.des, ii)
        print('>>>>>>>training : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')

        if args.model == 'Autoformer':
            model = Autoformer.Model(args).float()
        elif args.model == 'DLinear':
            model = DLinear.Model(args).float()
        else:
            model = TimeLLM.Model(args).float()

        path = os.path.join(args.checkpoints,
                            setting + '-' + args.model_comment)  # unique checkpoint saving path
        args.content = load_content(args)
        # if not os.path.exists(path) and accelerator.is_local_main_process:
        #     os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(verbose=True, patience=args.patience)

        trained_parameters = []
        for p in model.parameters():
            if p.requires_grad is True:
                trained_parameters.append(p)

        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

        if args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=args.pct_start,
                                                epochs=args.train_epochs,
                                                max_lr=args.learning_rate)

        criterion = nn.MSELoss()
        mae_metric = nn.L1Loss()

        # train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        #     train_loader, vali_loader, test_loader, model, model_optim, scheduler)

        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(args.train_epochs):
            iter_count = 0
            train_loss = []

            model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(device)
                # print("batch_x:", batch_x.device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                # print("batch_x_mark:", batch_x_mark.device)
                batch_y_mark = batch_y_mark.float().to(device)
                # print("batch_y_mark:", batch_y_mark.device)

                # decoder input
                # print("batch_y shape:", batch_y.shape) 
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                    device)
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                    device)
                # print("dec_inp:", dec_inp.device)

                # encoder - decoder
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        if args.output_attention:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if args.features == 'MS' else 0
                        outputs = outputs[:, -args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -args.pred_len:, f_dim:]
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, vali_mae_loss = vali(args, model, vali_data, vali_loader, criterion, mae_metric)
            test_loss, test_mae_loss = vali(args,  model, test_data, test_loader, criterion, mae_metric)
            print(
                "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}".format(
                    epoch + 1, train_loss, vali_loss, test_loss, test_mae_loss))

            early_stopping(vali_loss, model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if args.lradj != 'TST':
                if args.lradj == 'COS':
                    scheduler.step()
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                else:
                    if epoch == 0:
                        args.learning_rate = model_optim.param_groups[0]['lr']
                        print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, args, printout=True)

            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
else:
    ii = 0
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des,
        ii)
    setting = setting + '-' + args.model_comment
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    test_data, test_loader = data_provider(args, 'test')
    print("loading model")
    model = TimeLLM.Model(args).float()
    print((os.path.join('./checkpoints/' + setting, 'checkpoint')))
    # model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint')))
    model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint'), map_location='cuda:0'))

    # model.load_state_dict(torch.load(os.path.join('/raid/users/wx/时间序列/Time-LLM-main (2)/Time-LLM-main/checkpoints/long_term_forecast_ETTh1_512_96_TimeLLM_ETTh1_ftM_sl512_ll48_pl96_dm32_nh8_el2_dl1_df128_fc3_ebtimeF_Exp_0-TimeLLM-ETTh1',
    #                                               'checkpoint'), map_location='cuda'))
    # 获取当前日期
    current_date = datetime.now().strftime("%Y-%m-%d-%H-%M")
    preds = []
    trues = []
    folder_path = f'./test_results/{current_date}-{setting}/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    print("模型加载的位置")
    print(next(model.parameters()).device)
    model.to(device)
    print(next(model.parameters()).device)
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
            device = 'cuda'
            print(device)
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
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

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()
            # 可能有问题
            # if test_data.scale:
            #     shape = outputs.shape
            #     outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
            #     batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

            outputs = outputs[:, :, f_dim:]
            batch_y = batch_y[:, :, f_dim:]

            pred = outputs
            true = batch_y

            preds.append(pred)
            trues.append(true)
            if i % 20 == 0:
                input = batch_x.detach().cpu().numpy()
                # if test_data.scale and args.inverse:
                #     shape = input.shape
                #     input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
            # break
    preds = np.array(preds)
    trues = np.array(trues)
    print('test shape:', preds.shape, trues.shape)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('test shape:', preds.shape, trues.shape)

    # result save
    folder_path = f'./results/{current_date}-{setting}/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print('mse:{}, mae:{}'.format(mse, mae))
    f = open("result_long_term_forecast.txt", 'a')
    f.write(setting + "  \n")
    f.write('mse:{}, mae:{}'.format(mse, mae))
    f.write('\n')
    f.write('\n')
    f.close()
    torch.cuda.empty_cache()
# accelerator.wait_for_everyone()
# if accelerator.is_local_main_process:
#     path = './checkpoints'  # unique checkpoint saving path
#     del_files(path)  # delete checkpoint files
#     accelerator.print('success delete checkpoints')