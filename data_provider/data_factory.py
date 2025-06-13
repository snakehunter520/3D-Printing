from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, Dataset_3D
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# 新增自定义collate函数
def custom_collate_3d(batch):
    """处理3D数据集变长序列的填充"""
    # 解构批次数据：features, ot, time_stamp, status_desc
    features_list, ot_list, time_stamp_list, status_desc = zip(*batch)
    
    # 对变长序列进行填充
    padded_features = pad_sequence(features_list, batch_first=True, padding_value=0)
    padded_ot = pad_sequence(ot_list, batch_first=True, padding_value=0)
    padded_time = pad_sequence(time_stamp_list, batch_first=True, padding_value=0)
    
    return padded_features, padded_ot, padded_time, list(status_desc)

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'ECL': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Weather': Dataset_Custom,
    'm4': Dataset_M4,
    'retraction': Dataset_Custom,
    '3D': Dataset_3D  # 标记需要特殊处理的数据集
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    # 动态配置collate_fn
    collate_fn = None
    if args.data == '3D':  # 仅为3D数据集指定自定义collate
        collate_fn = custom_collate_3d

    # 通用参数配置
    common_params = {
        'batch_size': args.batch_size,
        'shuffle': flag != 'test',
        'num_workers': args.num_workers,
        'drop_last': flag == 'train',
        'collate_fn': collate_fn  # 注入自定义处理函数
    }

    # 数据集初始化
    if args.data == 'm4':
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=args.freq,
            seasonal_patterns=args.seasonal_patterns
        )
        common_params['drop_last'] = False  # M4特殊处理
    elif args.data == '3D':
        # 3D数据集需要传递额外参数
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=args.freq,
            percent=percent,
            seasonal_patterns=args.seasonal_patterns,
            task_split_threshold=3600  # 确保传递新增参数
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=args.freq,
            percent=percent,
            seasonal_patterns=args.seasonal_patterns
        )

    return data_set, DataLoader(data_set, **common_params)