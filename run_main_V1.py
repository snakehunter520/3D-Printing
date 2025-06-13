import argparse
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import numpy as np
import os
import time
import random
from datetime import datetime

# Local imports
from utils.tools import EarlyStopping, adjust_learning_rate, visual, load_content, vali
from utils.metrics import metric, classification_metrics  # 新增分类指标
from models import Autoformer, DLinear, TimeLLM
from data_provider.data_factory import data_provider

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

parser = argparse.ArgumentParser(description='Time-LLM for 3D Printing Defect Detection')

# 新增分类任务参数
parser.add_argument('--num_classes', type=int, default=4, help='number of anomaly classes')
parser.add_argument('--class_names', type=str, default='adhesion,clogging,bed_temp,motor',
                   help='comma-separated class names')

# 修改基础参数
# basic config
parser.add_argument(
    '--task_name',
    type=str,
    required=True,
    choices=['long_term_forecast', 'anomaly_classification', 'layer_prediction'],
    help='task name, options:[long_term_forecast, layer_prediction, anomaly_classification]'
)
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
parser.add_argument('--features', type=str, default='MS',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature for classification')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=128, help='input sequence length')
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
parser.add_argument('--patch_len', type=int, default=4, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model') # LLAMA, GPT2, BERT
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
# 保留原有其他参数...
# ... [保持原有参数定义不变] ...

args = parser.parse_args()

# 固定随机种子
fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

def classification_validate(model, loader, criterion, device):
    """分类任务验证函数"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        # 修改点：使用*rest接收多余元素
        for batch in loader:
            batch_x = batch[0].float().to(device)
            batch_y = batch[1].long().flatten().to(device)
            batch_x_mark = batch[2] if len(batch) > 2 else None
            
            outputs = model(batch_x, batch_x_mark, None, None)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    metrics = classification_metrics(all_labels, all_preds)
    return total_loss/len(loader), metrics

if args.is_training:
    for ii in range(args.itr):
        # 根据任务类型调整输出维度
        if args.task_name in ['anomaly_classification', 'layer_prediction']:
            args.c_out = args.num_classes
            args.pred_len = 1  # 分类任务预测长度固定为1

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
        
        # 数据加载（自动适配分类任务）
        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')

        # 模型初始化
        if args.model == 'TimeLLM':
            model = TimeLLM.Model(args).to(device)
            print(f"Initialized TimeLLM with {sum(p.numel() for p in model.parameters()):,} parameters")
        else:
            raise NotImplementedError(f"Model {args.model} not supported for classification")

        path = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
        args.content = load_content(args)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=args.patience,
            verbose=True,
            mode='max' if args.task_name in ['anomaly_classification', 'layer_prediction'] else 'min'
        )

        # 优化器设置
        trained_parameters = [p for p in model.parameters() if p.requires_grad]
        model_optim = optim.AdamW(trained_parameters, lr=args.learning_rate, weight_decay=1e-4)
        
        # 学习率调度
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                        steps_per_epoch=train_steps,
                                        pct_start=args.pct_start,
                                        epochs=args.train_epochs,
                                        max_lr=args.learning_rate)

        # 损失函数根据任务类型选择
        if args.task_name in ['anomaly_classification', 'layer_prediction']:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        # 混合精度训练
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

        for epoch in range(args.train_epochs):
            iter_count = 0
            train_loss = []
            correct = 0
            total = 0
            
            model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
                iter_count += 1
                model_optim.zero_grad()

                # 数据移动到设备
                batch_x = batch_x.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                
                # 分类任务数据处理
                if args.task_name in ['anomaly_classification', 'layer_prediction']:
                    batch_y = batch_y.long().flatten().to(device)  # 转换为长整型并展平
                    dec_inp = None  # 分类任务不需要decoder输入
                else:
                    batch_y = batch_y.float().to(device)
                    # 构造decoder输入（保持原有预测任务逻辑）
                    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(device)
                    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

                # 混合精度作用域
                with torch.cuda.amp.autocast(enabled=args.use_amp):
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    # 分类任务输出处理
                    if args.task_name in ['anomaly_classification', 'layer_prediction']:
                        print(outputs.shape)
                        print(batch_y.shape)

                        loss = criterion(outputs, batch_y)
                    else:
                        f_dim = -1 if args.features == 'MS' else 0
                        outputs = outputs[:, -args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
                        loss = criterion(outputs, batch_y)

                # 反向传播
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()

                # 统计训练指标
                train_loss.append(loss.item())
                
                # 分类任务统计准确率
                if args.task_name in ['anomaly_classification', 'layer_prediction']:
                    preds = outputs.argmax(dim=1)
                    correct += (preds == batch_y).sum().item()
                    total += batch_y.size(0)

                # 日志打印间隔
                if (i + 1) % 100 == 0:
                    speed = (time.time() - epoch_time) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    
                    log_msg = "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item())
                    if args.task_name in ['anomaly_classification', 'layer_prediction']:
                        log_msg += f" | acc: {correct/total:.2%}"
                    print(log_msg)

                    iter_count = 0
                    epoch_time = time.time()

            # 验证阶段
            if args.task_name in ['anomaly_classification', 'layer_prediction']:
                val_loss, val_metrics = classification_validate(model, vali_loader, criterion, device)
                log_msg = (
                    f"Epoch: {epoch + 1} | Train Loss: {np.average(train_loss):.4f} | Acc: {correct/total:.2%} | "
                    f"Val Acc: {val_metrics['accuracy']:.2%} | Val F1: {val_metrics['f1']:.4f}"
                )
                early_stopping(-val_metrics['f1'], model, path)  # 监控F1分数
            else:
                val_loss, val_mae = vali(args, model, vali_data, vali_loader, criterion, nn.L1Loss())
                log_msg = "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}".format(
                    epoch + 1, np.average(train_loss), val_loss, val_loss, val_mae)
                early_stopping(val_loss, model, path)

            print(log_msg)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 学习率调整
            if args.lradj == 'COS':
                scheduler.step()
            elif args.lradj == 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, args)

else:
    # 测试代码（保持原有结构，增加分类任务支持）
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
    
    # 加载测试数据
    test_data, test_loader = data_provider(args, 'test')
    
    # 加载模型
    model = TimeLLM.Model(args).to(device)
    model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint'), map_location=device))
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
    # 测试逻辑（根据任务类型分支）
    if args.task_name in ['anomaly_classification', 'layer_prediction']:
        test_loss, test_metrics = classification_validate(model, test_loader, nn.CrossEntropyLoss(), device)
        print(f"\nFinal Test Results - Acc: {test_metrics['accuracy']:.2%} | F1: {test_metrics['f1']:.4f}")
    else:
        # 保持原有预测任务测试逻辑
        preds, trues = [], []
        model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # ... [保持原有预测任务测试代码] ...
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
            # 计算并保存预测结果
        mae, mse = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

