import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
import warnings
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler


warnings.filterwarnings('ignore')

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        # self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    # def __read_data__(self):
    #     self.scaler = StandardScaler()
    #     df_raw = pd.read_csv(os.path.join(self.root_path,
    #                                       self.data_path))

    #     border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
    #     border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

    #     border1 = border1s[self.set_type]
    #     border2 = border2s[self.set_type]

        # if self.set_type == 0:
        #     border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        # if self.features == 'M' or self.features == 'MS':
        #     cols_data = df_raw.columns[1:]
        #     df_data = df_raw[cols_data]
        # elif self.features == 'S':
        #     df_data = df_raw[[self.target]]

        # if self.scale:
        #     train_data = df_data[border1s[0]:border2s[0]]
        #     self.scaler.fit(train_data.values)
        #     data = self.scaler.transform(df_data.values)
        # else:
        #     data = df_data.values

        # df_stamp = df_raw[['date']][border1:border2]
        # df_stamp['date'] = pd.to_datetime(df_stamp.date)
        # if self.timeenc == 0:
        #     df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        #     df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        #     df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        #     df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        #     data_stamp = df_stamp.drop(['date'], 1).values
        # elif self.timeenc == 1:
        #     data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
        #     data_stamp = data_stamp.transpose(1, 0)

        # self.data_x = data[border1:border2]
        # self.data_y = data[border1:border2]
        # self.data_stamp = data_stamp
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # 动态计算数据集分割点
        total_length = len(df_raw)
        train_ratio = 0.7  # 训练集比例
        val_ratio = 0.1    # 验证集比例
        test_ratio = 0.2   # 测试集比例
        
        # 计算各分区的结束位置
        train_end = int(total_length * train_ratio)
        val_end = train_end + int(total_length * val_ratio)
        test_end = total_length

        # 计算边界点，确保不小于0
        border1s = [
            0,
            max(0, train_end - self.seq_len),
            max(0, val_end - self.seq_len)
        ]
        border2s = [train_end, val_end, test_end]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 调整训练集边界（考虑percent参数）
        if self.set_type == 0:
            new_border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len
            new_border2 = min(new_border2, border2)  # 不超过原始边界
            new_border2 = max(new_border2, border1 + self.seq_len)  # 确保最小长度
            border2 = new_border2

        # 确保边界有效性
        border1 = max(0, border1)
        border2 = max(border1, border2)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)




class Dataset_3D(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                features='MS', data_path='all.csv',
                target='OT', scale=True, timeenc=0, freq='h', 
                pred_len=1, percent=100, seasonal_patterns=None,
                task_split_threshold=3600):
        
        # 配置参数
        self.seq_len = size[0] if size else 120
        self.pred_len = pred_len
        self.label_len = 0
        self.task_split_threshold = task_split_threshold
        self.flag = flag
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path

        # 列定义（仅用于原始数据读取）
        self.raw_columns = [
            'time', 'timestamp', 'aX(m/s²)', 'aY(m/s²)', 'aZ(m/s²)',
            'nozzle_tem', 'bed_tem', 'THE(°C)', 'current_layer', 
            'total_layers', 'OT', 'task_id'
        ]
        
        # 特征列索引（根据处理后数据矩阵的列顺序）
        self.time_idx = 0
        self.vib_start = 1     # 时间戳后接3列振动数据
        self.temp_start = 4   # 振动后接3列温度数据
        self.layer_start = 7   # 温度后接2列层数据
        self.ot_idx = 9        # OT列位置
        self.task_idx = 10     # task_id列位置

        # 初始化数据结构
        self.layer_indices = []  # 存储各层的起止索引 [(start1, end1), ...]
        self.layer_metadata = [] 
        self.scalers = {}
        
        # 数据加载流程
        self.__read_data__()

    def __read_data__(self):
        # 1. 加载原始数据
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # 2. 时间处理
        df_raw['time'] = pd.to_datetime(df_raw['time'], format='%Y-%m-%d-%H-%M-%S')
        df_raw['timestamp'] = df_raw['time'].astype('int64') // 10**9
        
        # 3. 生成任务ID
        df_raw = self._generate_task_ids(df_raw)
        
        # 4. 划分数据集
        self._split_data_by_task(df_raw)
        
        # 5. 分层标准化
        if self.scale:
            self._task_based_scaling()
        else:
            # 未标准化时直接构建数据矩阵
            self.data = np.hstack([
                df_raw[['timestamp']].values.astype(np.float32),
                df_raw[self.raw_columns[2:5]].values.astype(np.float32),  # 振动
                df_raw[self.raw_columns[5:8]].values.astype(np.float32),  # 温度
                df_raw[['current_layer', 'total_layers']].values.astype(np.int32),
                df_raw[['OT', 'task_id']].values.astype(np.float32)
            ])
        
        # 6. 构建层索引
        self._build_layer_indices()
        
        # 7. 特征工程
        self._prepare_features()

    def _generate_task_ids(self, df):
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['task_id'] = 0
        current_task = 0
        last_timestamp = df.at[0, 'timestamp']

        for i in range(1, len(df)):
            time_gap = df.at[i, 'timestamp'] - last_timestamp
            if time_gap > 900:
                current_task += 1
                df.at[i, 'current_layer'] = 0  # 新任务重置层数

            last_timestamp = df.at[i, 'timestamp']
            df.at[i, 'task_id'] = current_task

        # 验证层连续性
        for task_id in df['task_id'].unique():
            task_data = df[df['task_id'] == task_id]
            if task_data['current_layer'].max() > 206:
                raise ValueError(f"Task {task_id} layer overflow")
        return df

    def _split_data_by_task(self, df_raw):
        task_ids = df_raw['task_id'].unique()
        np.random.shuffle(task_ids)
        
        # 动态计算分割数量
        total_tasks = len(task_ids)
        train_end = int(total_tasks * 0.7)
        val_end = train_end + int(total_tasks * 0.2)
        
        if self.flag == 'train':
            selected = task_ids[:train_end]
        elif self.flag == 'val':
            selected = task_ids[train_end:val_end]
        else:
            selected = task_ids[val_end:]
        
        # 保证至少一个任务
        if len(selected) == 0:
            selected = [task_ids[0]]
            print(f"Warning: {self.flag} set using first task")
            
        self.df_data = df_raw[df_raw['task_id'].isin(selected)].sort_values('timestamp')

    def _task_based_scaling(self):
        scaled_features = []
        self.scalers = {}

        for task_id, group in self.df_data.groupby('task_id'):
            # 振动标准化
            vib_scaler = StandardScaler()
            scaled_vib = vib_scaler.fit_transform(group[self.raw_columns[2:5]])
            
            # 温度归一化
            temp_scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled_temp = temp_scaler.fit_transform(group[self.raw_columns[5:8]])
            
            # 保存scaler
            self.scalers[task_id] = {
                'vibration': vib_scaler,
                'temperature': temp_scaler
            }
            
            # 构建数据矩阵
            task_data = np.hstack([
                group[['timestamp']].values.astype(np.float32),
                scaled_vib.astype(np.float32),
                scaled_temp.astype(np.float32),
                group[['current_layer', 'total_layers']].values.astype(np.int32),
                group[['OT', 'task_id']].values.astype(np.float32)
            ])
            scaled_features.append(task_data)
        
        self.data = np.vstack(scaled_features)

    def _build_layer_indices(self):
        current_layer = -1
        start_idx = 0
        current_task = -1

        for i in range(len(self.data)):
            # 直接通过固定索引获取层和任务信息
            layer = int(self.data[i, self.layer_start])
            task = int(self.data[i, self.task_idx])

            if layer != current_layer or task != current_task:
                if current_layer != -1:
                    self.layer_indices.append((start_idx, i))
                    self.layer_metadata.append({
                        'task_id': current_task,
                        'layer_num': current_layer,
                        'total_layers': int(self.data[i-1, self.layer_start+1])
                    })
                current_layer = layer
                current_task = task
                start_idx = i
        
        # 处理最后一个层
        if current_layer != -1:
            self.layer_indices.append((start_idx, len(self.data)))
            self.layer_metadata.append({
                'task_id': current_task,
                'layer_num': current_layer,
                'total_layers': int(self.data[-1, self.layer_start+1])
            })

    def _prepare_features(self):
        # 振动量级特征
        vib_data = self.data[:, self.vib_start:self.vib_start+3]
        vib_magnitude = np.linalg.norm(vib_data, axis=1)[:, None]  # 保持二维
        
        # 温度差异特征
        nozzle_temp = self.data[:, self.temp_start]
        bed_temp = self.data[:, self.temp_start+1]
        temp_diff = (nozzle_temp - bed_temp)[:, None]
        
        # 层进度特征
        current_layer = self.data[:, self.layer_start]
        total_layers = self.data[:, self.layer_start+1]
        layer_progress = (current_layer / (total_layers + 1e-6))[:, None]
        
        # 合并时间特征
        self.data_stamp = np.hstack([
            vib_magnitude.astype(np.float32),
            temp_diff.astype(np.float32),
            layer_progress.astype(np.float32)
        ])

    def __len__(self):
        return len(self.layer_indices)

    def __getitem__(self, index):
        start, end = self.layer_indices[index]
        metadata = self.layer_metadata[index]
        
        # 提取特征数据
        features = self.data[start:end, :self.ot_idx]  # 排除OT和task_id
        ot = self.data[start:end, self.ot_idx]
        
        # 时间特征
        time_feat = self.data_stamp[start:end]
        
        # 温度逆变换
        task_id = metadata['task_id']
        temp_scaler = self.scalers[task_id]['temperature']
        nozzle_temp = temp_scaler.inverse_transform(
            features[:, self.temp_start:self.temp_start+3]
        )[:, 0]  # 取第一个温度值

        return (
            torch.FloatTensor(features),
            torch.FloatTensor(ot),
            torch.FloatTensor(time_feat),
            f"Task:{task_id} Layer:{metadata['layer_num']}/{metadata['total_layers']} Samples:{end-start}"
        )

    def inverse_transform(self, data, feature_type='vibration', task_id=None):
        if task_id not in self.scalers:
            raise ValueError(f"Scaler for task {task_id} not found")
        return self.scalers[task_id][feature_type].inverse_transform(data)


class Dataset_3D_V1(Dataset):
    """
    root_path是数据集所在的根目录，flag表示数据集的类型（训练、测试或验证），size是序列的长度，
    features是使用的特征类型，data_path是数据文件的路径，target是目标变量，
    scale表示是否对数据进行标准化，timeenc表示是否使用时间编码，
    freq是时间序列的频率，percent是训练集的比例，seasonal_patterns是季节性模式。
    这段代码可能会有单变量输入的问题
    """
    def __init__(self, root_path, flag='train', size=None,
                features='MS', data_path=None,
                target='OT', scale=True, timeenc=0, freq='h', percent=100,
                seasonal_patterns=None):
        # 如果size没有指定，则设置默认的序列长度、标签长度和预测长度
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        # 确保flag的值是'train'、'test'或'val'之一
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        # 设置特征类型、目标变量、是否进行数据标准化等参数
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        # 设置数据集的根目录和文件路径
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()


        # 设置输入特征的维度
        self.enc_in = self.data_x.shape[-1]
        # 计算总长度
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1
    def __read_data__(self):
        # 创建一个StandardScaler对象，用于数据标准化
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path), encoding='gb18030')
        
        df_raw.columns = df_raw.columns.str.strip()  # 去除列名首尾空格
        df_raw.columns = df_raw.columns.str.replace('\u200b', '')  # 去除零宽度空格
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # 列顺序调整：将'time'作为日期列，其他特征保留，目标列在最后
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('time')  # 原代码中的'date'改为'time'
        df_raw = df_raw[['time'] + cols + [self.target]]  # 确保列顺序为[time, features..., OT]
        # 时间解析（处理毫秒格式）
        df_raw['time'] = pd.to_datetime(df_raw['time'], format='%Y-%m-%d-%H-%M-%S')
        
        
        # 计算训练集、测试集和验证集的大小
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        # 计算各个数据集的起始和结束索引
        # 计算各个数据集的起始索引
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        # 计算各个数据集的结束索引
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 如果是训练集，则调整border2的值
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        # 根据特征类型，选择使用哪些特征
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # 如果进行数据标准化，则对数据进行标准化处理
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 时间特征生成（适配新时间格式）
        df_stamp = df_raw[['time']][border1:border2]
        df_stamp['time'] = pd.to_datetime(df_stamp['time'], format='%Y-%m-%d-%H-%M-%S')
        
        if self.timeenc == 0:
            # 提取更精细的时间特征（如秒、毫秒）
            df_stamp['second'] = df_stamp.time.dt.second
            
            data_stamp = df_stamp.drop(['time'], axis=1).values
        elif self.timeenc == 1:
            # 使用自定义时间编码函数
            # data_stamp = time_features(df_stamp['time'], freq=self.freq)
            # 使用pd.DatetimeIndex转换
            dates = pd.DatetimeIndex(df_stamp['time'].values)
            data_stamp = time_features(dates, freq=self.freq)  # 传递DatetimeIndex而非Series
            data_stamp = data_stamp.transpose(1, 0)

        # 根据边界设置输入数据和输出数据
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        # 设置时间特征数据
        self.data_stamp = data_stamp         
        
    # 下面的__getitem__方法和__len__可能有一点的问题，下面是deepseek给的修改后的代码
    def __getitem__(self, index):
        # 输入序列的起始和结束索引
        s_begin = index
        s_end = s_begin + self.seq_len

        # 输出序列的起始和结束索引
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # 输入：所有特征
        seq_x = self.data_x[s_begin:s_end, :]  # 所有列作为输入
        # 输出：目标变量（最后一列）
        seq_y = self.data_y[r_begin:r_end, -1:]  # 仅OT列

        # 时间特征
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    


class Dataset_3D_V4(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                features='MS', data_path='all.csv',
                target='OT', scale=True, timeenc=0, freq='h', 
                pred_len=1, percent=100, seasonal_patterns=None,
                task_split_threshold=3600):  # 新增任务分割阈值（1小时）
        
        self.seq_len = size[0] if size else 120
        self.pred_len = pred_len
        self.label_len = 0
        self.task_split_threshold = task_split_threshold
        
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        
        # 新增task_id列
        self.raw_columns = [
            'time', 'timestamp', 
            'aX(m/s²)', 'aY(m/s²)', 'aZ(m/s²)',
            'nozzle_tem', 'bed_tem', 'THE(°C)',
            'current_layer', 'total_layers',
            'OT', 'task_id'  # 新增任务ID列
        ]
        self.vibration_cols = ['aX(m/s²)', 'aY(m/s²)', 'aZ(m/s²)']
        self.temp_cols = ['nozzle_tem', 'bed_tem', 'THE(°C)']
        self.layer_cols = ['current_layer', 'total_layers']
        self.time_cols = ['timestamp']
        
        self.__read_data__()

    def __read_data__(self):
        # 读取原始数据
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # 预处理时间列
        df_raw['time'] = pd.to_datetime(df_raw['time'], format='%Y-%m-%d-%H-%M-%S')
        df_raw['timestamp'] = df_raw['time'].astype('int64') // 10**9
        
        # 自动生成任务ID（核心修改）
        df_raw = self._generate_task_ids(df_raw)
        
        # 按任务划分数据集（核心修改）
        self._split_data_by_task(df_raw)
        
        # 分层标准化（每个任务单独标准化）
        if self.scale:
            self._task_based_scaling()
        
        # 特征工程
        self._prepare_features()
        
    def _generate_task_ids(self, df):
        """精确的打印任务分割逻辑（15分钟间隔）"""
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['task_id'] = 0
        current_task = 0
        last_timestamp = df.at[0, 'timestamp']

        for i in range(1, len(df)):
            time_gap = df.at[i, 'timestamp'] - last_timestamp

            # 关键修改：严格15分钟间隔判断（900秒）
            if time_gap > 900:  # 15分钟 = 900秒
                current_task += 1
                # 重置层数为0（新任务开始）
                df.at[i, 'current_layer'] = 0

            last_timestamp = df.at[i, 'timestamp']
            df.at[i, 'task_id'] = current_task

        # 验证层数连续性
        for task_id in df['task_id'].unique():
            task_data = df[df['task_id'] == task_id]
            max_layer = task_data['current_layer'].max()
            if max_layer > 206:
                raise ValueError(f"任务 {task_id} 层数异常：max_layer={max_layer}")

        print(f"检测到有效打印任务数：{current_task + 1}")
        return df

    def _split_data_by_task(self, df_raw):
        """更鲁棒的任务划分"""
        task_ids = df_raw['task_id'].unique()
        np.random.shuffle(task_ids)

        # 强制每个数据集至少有1个任务
        min_tasks = 1
        num_tasks = max(min_tasks, len(task_ids))

        # 划分比例调整为 7:2:1
        train_ratio, val_ratio = 0.6, 0.2
        num_train = max(min_tasks, int(num_tasks * train_ratio))
        num_val = max(min_tasks, int(num_tasks * val_ratio))
        num_test = max(min_tasks, num_tasks - num_train - num_val)

        # 任务分配
        train_tasks = task_ids[:num_train]
        val_tasks = task_ids[num_train:num_train+num_val]
        test_tasks = task_ids[num_train+num_val:]

        # 根据flag选择任务
        if self.flag == 'train':
            selected_tasks = train_tasks
        elif self.flag == 'val':
            selected_tasks = val_tasks
        else:
            selected_tasks = test_tasks

        # 最终保障：至少包含一个任务
        if len(selected_tasks) == 0:
            selected_tasks = [task_ids[0]]
            print(f"⚠️ {self.flag}集无任务，自动分配第一个任务")

        self.df_data = df_raw[df_raw['task_id'].isin(selected_tasks)].sort_values('timestamp')
        print(f"数据集 {self.flag} 包含 {len(selected_tasks)} 个任务，{len(self.df_data)} 行数据")

    def _task_based_scaling(self):
        self.scalers = {}
        scaled_features = []

        # 获取实际存在的任务ID
        valid_tasks = self.df_data['task_id'].unique()
        print(f"需要标准化的任务ID：{valid_tasks}")

        for task_id, group in self.df_data.groupby('task_id'):
            if group.empty:
                raise ValueError(f"任务 {task_id} 数据为空！")

            # 振动标准化
            vib_scaler = StandardScaler()
            scaled_vib = vib_scaler.fit_transform(group[self.vibration_cols])

            # 温度归一化
            temp_scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled_temp = temp_scaler.fit_transform(group[self.temp_cols])

            # 层数数据保持原始值
            layer_data = group[self.layer_cols].values.astype(np.int32)

            # 保存scaler
            self.scalers[task_id] = {
                'vibration': vib_scaler,
                'temperature': temp_scaler
            }

            # 合并数据
            scaled_group = np.hstack([
                group[self.time_cols].values.astype(np.float32),
                scaled_vib.astype(np.float32),
                scaled_temp.astype(np.float32),
                layer_data,
                group[['OT', 'task_id']].values.astype(np.float32)
            ])
            scaled_features.append(scaled_group)

        # 检查scaler完整性
        for task_id in self.df_data['task_id'].unique():
            if task_id not in self.scalers:
                raise KeyError(f"任务 {task_id} 的scaler未生成！")

        self.data = np.vstack(scaled_features)
        
    def _prepare_features(self):
        """准备时间特征和其他派生特征"""
        # 获取特征列索引
        self.feature_cols = self.time_cols + self.vibration_cols + self.temp_cols + self.layer_cols
        self.stamp_cols = ['time']
        
        # 准备时间特征
        self.df_stamp = pd.DataFrame({
            'time': pd.to_datetime(self.df_data['time']),
            'task_id': self.df_data['task_id']
        })
        self._add_time_features()
        
    def _add_time_features(self):
        """添加时间相关特征"""
        # 振动量级特征
        vib_start = self.feature_cols.index(self.vibration_cols[0])
        vib_data = self.data[:, vib_start:vib_start+3]
        self.df_stamp['vib_magnitude'] = np.linalg.norm(vib_data, axis=1)
        
        # 温度差异特征
        nozzle_idx = self.feature_cols.index('nozzle_tem')
        bed_idx = self.feature_cols.index('bed_tem')
        self.df_stamp['temp_diff'] = self.data[:, nozzle_idx] - self.data[:, bed_idx]
        
        # 层进度特征
        current_layer_idx = self.feature_cols.index('current_layer')
        total_layer_idx = self.feature_cols.index('total_layers')
        self.df_stamp['layer_progress'] = self.data[:, current_layer_idx] / \
                                        (self.data[:, total_layer_idx] + 1e-6)
        
        # 时间编码
        if self.timeenc == 1:
            self.df_stamp['hour'] = self.df_stamp.time.dt.hour
            self.df_stamp['minute'] = self.df_stamp.time.dt.minute
        
        self.data_stamp = self.df_stamp.drop('time', axis=1).values

    def __getitem__(self, index):
        s = index
        e = s + self.seq_len
        y_end = e + self.pred_len

        # 获取当前窗口所属的任务ID
        task_id = int(self.data[e-1, -1])  # 最后一列是task_id

        # 获取对应任务的scaler（不再需要层数scaler）
        scalers = self.scalers[task_id]

        # 直接获取原始层数（无需逆变换）
        current_layer = int(self.data[e-1, self.feature_cols.index('current_layer')])
        total_layers = int(self.data[e-1, self.feature_cols.index('total_layers')])

        # 逆变换温度数据
        nozzle_temp_scaled = self.data[e-1, self.feature_cols.index('nozzle_tem')]
        nozzle_temp = scalers['temperature'].inverse_transform(
            [[nozzle_temp_scaled, 0, 0]])[0, 0]  # 假设其他温度值为0

        # 构建状态描述
        status_desc = f"Layer {current_layer}/{total_layers} | Nozzle: {nozzle_temp:.1f}°C"

        return (
            torch.FloatTensor(self.data[s:e, :-2]),  # 排除OT和task_id
            torch.LongTensor(self.data[e:y_end, -2]),  # OT列
            torch.FloatTensor(self.data_stamp[s:e]),
            torch.FloatTensor(self.data_stamp[e:y_end]),
            status_desc
        )

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data, feature_type='vibration', task_id=None):
        """支持按任务进行逆变换"""
        if task_id is None:
            raise ValueError("逆变换需要指定task_id")
        return self.scalers[task_id][feature_type].inverse_transform(data)



class Dataset_3D_V3(Dataset):
    # 没有按照任务划分数据集
    def __init__(self, root_path, flag='train', size=None, 
                features='MS', data_path='all.csv',
                target='OT', scale=True, timeenc=0, freq='h', 
                pred_len=1, percent=100, seasonal_patterns=None):
        
        self.seq_len = size[0] if size else 120
        self.pred_len = pred_len
        self.label_len = 0
        
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        
        self.raw_columns = [
            'time', 'timestamp', 
            'aX(m/s²)', 'aY(m/s²)', 'aZ(m/s²)',
            'nozzle_tem', 'bed_tem', 'THE(°C)',
            'current_layer', 'total_layers',
            'OT'
        ]
        self.vibration_cols = ['aX(m/s²)', 'aY(m/s²)', 'aZ(m/s²)']
        self.temp_cols = ['nozzle_tem', 'bed_tem', 'THE(°C)']
        self.layer_cols = ['current_layer', 'total_layers']
        self.time_cols = ['timestamp']  # 仅使用数值型时间戳
        
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # 处理时间列
        df_raw['time'] = pd.to_datetime(df_raw['time'], format='%Y-%m-%d-%H-%M-%S')
        df_raw['timestamp'] = df_raw['time'].astype('int64') // 10**9  # 转换为Unix时间戳
        
        # 数据划分
        num_all = len(df_raw)
        num_train = int(num_all * 0.7)
        num_test = int(num_all * 0.2)
        num_val = num_all - num_train - num_test
        
        border1s = [0, num_train, num_train + num_val]
        border2s = [num_train, num_train + num_val, num_all]
        
        self.border1 = border1s[['train', 'val', 'test'].index(self.flag)]
        self.border2 = border2s[['train', 'val', 'test'].index(self.flag)]
        
        df_data = df_raw.iloc[self.border1:self.border2].reset_index(drop=True)
        
        # 数据验证
        assert not df_data.empty, "分割后数据为空！"
        assert df_data['total_layers'].nunique() == 1, "存在不一致的总层数"
        
        # 分层标准化
        self.scalers = {}
        vib_scaler = StandardScaler()
        df_data[self.vibration_cols] = vib_scaler.fit_transform(df_data[self.vibration_cols])
        
        temp_scaler = MinMaxScaler(feature_range=(-1, 1))
        df_data[self.temp_cols] = temp_scaler.fit_transform(df_data[self.temp_cols])
        
        layer_scaler = MinMaxScaler()
        df_data[self.layer_cols] = layer_scaler.fit_transform(df_data[self.layer_cols])
        
        self.scalers = {
            'vibration': vib_scaler,
            'temperature': temp_scaler,
            'layers': layer_scaler
        }

        # 特征和标签分离
        self.feature_cols = self.time_cols + self.vibration_cols + self.temp_cols + self.layer_cols
        self.data = df_data[self.feature_cols].values.astype(np.float32)  # 确保数据类型为浮点
        self.target_data = df_data[[self.target]].values.astype(int)
        
        # 时间特征处理（保留原始时间列用于生成特征）
        self.df_stamp = df_data[['time']].reset_index(drop=True)
        self._add_time_features()
        
        assert len(self.data) == len(self.df_stamp), \
            f"特征矩阵({len(self.data)})和时间戳({len(self.df_stamp)})维度不匹配！"

    def _add_time_features(self):
        df_stamp = pd.DataFrame(index=self.df_stamp.index)
        df_stamp['time'] = pd.to_datetime(self.df_stamp['time'])
        
        # 添加打印过程特征
        df_stamp['layer_progress'] = self.data[:, self.feature_cols.index('current_layer')]
        df_stamp['total_progress'] = self.data[:, self.feature_cols.index('total_layers')]
        df_stamp['temp_diff'] = self.data[:, self.feature_cols.index('nozzle_tem')] - \
                            self.data[:, self.feature_cols.index('bed_tem')]
        
        # 确保振动数据为浮点型
        vib_start_idx = self.feature_cols.index(self.vibration_cols[0])
        vib_data = self.data[:, vib_start_idx:vib_start_idx+3].astype(np.float32)
        df_stamp['vib_magnitude'] = np.linalg.norm(vib_data, axis=1)
        
        # 时间编码
        if self.timeenc == 1:
            df_stamp['hour'] = df_stamp.time.dt.hour
            df_stamp['minute'] = df_stamp.time.dt.minute
        
        self.data_stamp = df_stamp.drop('time', axis=1).values

    def __getitem__(self, index):
        s = index
        e = s + self.seq_len
        y_start = e
        y_end = y_start + self.pred_len

        seq_x = self.data[s:e]
        seq_y = self.target_data[y_start:y_end].flatten()
        
        # 使用逆变换获取实际值
        current_layer_scaled = self.data[e-1, self.feature_cols.index('current_layer')]
        total_layers_scaled = self.data[e-1, self.feature_cols.index('total_layers')]
        current_layer = int(self.scalers['layers'].inverse_transform(
            [[current_layer_scaled, total_layers_scaled]])[0, 0])
        
        nozzle_temp_scaled = self.data[e-1, self.feature_cols.index('nozzle_tem')]
        nozzle_temp = self.scalers['temperature'].inverse_transform(
            [[nozzle_temp_scaled, 0, 0]])[0, 0]  # 假设其他温度值为0
        
        status_desc = f"Layer {current_layer}/206 | Nozzle: {nozzle_temp:.1f}°C"

        return (
            torch.FloatTensor(seq_x),
            torch.LongTensor(seq_y),
            torch.FloatTensor(self.data_stamp[s:e]),
            torch.FloatTensor(self.data_stamp[y_start:y_end]),
            status_desc
        )

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data, feature_type='vibration'):
        return self.scalers[feature_type].inverse_transform(data)


class Dataset_3D_V2(Dataset):
    """
    修改后的数据集类，专用于多步二分类预测
    主要修改点：
    1. 分离特征和标签，避免OT被标准化
    2. 输出未来pred_len步的OT标签
    3. 增强时间特征编码
    4. 数据划分防泄露
    """
    def __init__(self, root_path, flag='train', size=None,
                features='MS', data_path=None,
                target='OT', scale=True, timeenc=0, freq='h', 
                pred_len=1, percent=100, seasonal_patterns=None):
        # 时序参数设置
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[2] if size[2] is not None else pred_len
        
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        # 读取原始数据
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), 
                            encoding='gb18030')
        
        # 清洗列名
        df_raw.columns = df_raw.columns.str.strip().str.replace('\u200b', '')
        
        # ---- 特征列分离 ----
        # 排除时间相关列和标签列
        self.feature_cols = [col for col in df_raw.columns 
                        if col not in [self.target, 'time', 'timestamp']]
        self.label_col = self.target
        
        # ---- 时间解析 ----
        df_raw['time'] = pd.to_datetime(df_raw['time'], format='%Y-%m-%d-%H-%M-%S')
        
        # ---- 数据划分 ----
        # 按时间顺序划分，防止未来信息泄露
        num_all = len(df_raw)
        num_train = int(num_all * 0.7)
        num_test = int(num_all * 0.2)
        num_val = num_all - num_train - num_test
        
        border1s = [
            0,  # train start
            num_train - self.seq_len,  # val start
            num_train + num_val - self.seq_len  # test start
        ]
        border2s = [
            num_train,  # train end
            num_train + num_val,  # val end
            num_all  # test end
        ]
        
        # 根据flag选择边界
        self.border1 = border1s[['train', 'val', 'test'].index(self.flag)]
        self.border2 = border2s[['train', 'val', 'test'].index(self.flag)]
        
        # ---- 特征处理 ----
        # 仅对特征列进行标准化
        feature_data = df_raw[self.feature_cols].values
        label_data = df_raw[self.label_col].values.reshape(-1, 1)  # 保持二维结构
        
        if self.scale:
            # 仅用训练集数据拟合scaler
            train_features = feature_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_features)
            self.feature_data = self.scaler.transform(feature_data)
        else:
            self.feature_data = feature_data
        
        # 标签保持原始值（0/1）
        self.label_data = label_data
        
        # ---- 时间特征工程 ----
        self.df_stamp = df_raw[['time']].iloc[self.border1:self.border2]
        self._add_time_features()

    def _add_time_features(self):
        """添加丰富的时间特征"""
        df_stamp = self.df_stamp.copy()
        df_stamp['month'] = df_stamp.time.dt.month
        df_stamp['day'] = df_stamp.time.dt.day
        df_stamp['hour'] = df_stamp.time.dt.hour
        df_stamp['minute'] = df_stamp.time.dt.minute
        df_stamp['weekday'] = df_stamp.time.dt.weekday  # Monday=0, Sunday=6
        df_stamp['is_weekend'] = df_stamp.weekday.apply(lambda x: 1 if x >= 5 else 0)
        df_stamp['seconds'] = df_stamp.time.dt.hour * 3600 + \
                             df_stamp.time.dt.minute * 60 + \
                            df_stamp.time.dt.second
        
        if self.timeenc == 0:
            self.data_stamp = df_stamp.drop('time', axis=1).values
        elif self.timeenc == 1:
            # 使用自定义时间编码函数
            # data_stamp = time_features(df_stamp['time'], freq=self.freq)
            # 使用pd.DatetimeIndex转换
            dates = pd.DatetimeIndex(df_stamp['time'].values)
            data_stamp = time_features(dates, freq=self.freq)  # 传递DatetimeIndex而非Series
            self.data_stamp = data_stamp.transpose(1, 0)

    def __getitem__(self, index):
        # 实际有效索引范围
        valid_start = self.border1
        valid_end = self.border2 - self.seq_len - self.pred_len + 1
        
        # 计算绝对位置
        pos = valid_start + index
        
        # 输入窗口 [pos, pos+seq_len)
        seq_x = self.feature_data[pos:pos+self.seq_len]
        
        # 输出窗口 [pos+seq_len, pos+seq_len+pred_len)
        y_start = pos + self.seq_len
        y_end = y_start + self.pred_len
        seq_y = self.label_data[y_start:y_end]
        
        # 时间特征（取输入窗口的时间）
        stamp_x = self.data_stamp[pos:pos+self.seq_len]
        stamp_y = self.data_stamp[y_start:y_end]
        
        return (
            torch.FloatTensor(seq_x),         # 输入特征 [seq_len, num_features]
            torch.LongTensor(seq_y),          # 输出标签 [pred_len, 1] ← 关键修改点
            torch.FloatTensor(stamp_x),       # 输入时间特征 
            torch.FloatTensor(stamp_y)        # 输出时间特征
        )

    def __len__(self):
        return self.border2 - self.border1 - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """逆变换（适用于特征列）"""
        return self.scaler.inverse_transform(data)
    
class Dataset_Custom(Dataset):
    """
    root_path是数据集所在的根目录，flag表示数据集的类型（训练、测试或验证），size是序列的长度，
    features是使用的特征类型，data_path是数据文件的路径，target是目标变量，
    scale表示是否对数据进行标准化，timeenc表示是否使用时间编码，
    freq是时间序列的频率，percent是训练集的比例，seasonal_patterns是季节性模式。
    """
    def __init__(self, root_path, flag='train', size=None,
                features='S', data_path='ETTh1.csv',
                target='OT', scale=True, timeenc=0, freq='h', percent=100,
                seasonal_patterns=None):
        # 如果size没有指定，则设置默认的序列长度、标签长度和预测长度
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        # 确保flag的值是'train'、'test'或'val'之一
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        # 设置特征类型、目标变量、是否进行数据标准化等参数
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        # 设置数据集的根目录和文件路径
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()


        # 设置输入特征的维度
        self.enc_in = self.data_x.shape[-1]
        # 计算总长度
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        # 创建一个StandardScaler对象，用于数据标准化
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                        self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # 获取数据框的列名，移除目标变量和日期列
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        # 重新组织数据框的列
        df_raw = df_raw[['date'] + cols + [self.target]]
        # 计算训练集、测试集和验证集的大小
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        # 计算各个数据集的起始和结束索引
        # 计算各个数据集的起始索引
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        # 计算各个数据集的结束索引
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 如果是训练集，则调整border2的值
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        # 根据特征类型，选择使用哪些特征
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # 如果进行数据标准化，则对数据进行标准化处理
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 提取时间戳数据，将日期列转换为datetime类型
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        # 如果使用时间编码，则添加时间特征
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # 根据边界设置输入数据和输出数据
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        # 设置时间特征数据
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        # 计算特征ID
        feat_id = index // self.tot_len
        # 计算序列的开始索引
        s_begin = index % self.tot_len

        # 计算序列的结束索引
        s_end = s_begin + self.seq_len
        # 计算标签序列的开始索引
        r_begin = s_end - self.label_len
        # 计算标签序列的结束索引
        r_end = r_begin + self.label_len + self.pred_len

        # 提取输入序列
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        # 提取输出序列
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        # 提取输入序列的时间特征
        seq_x_mark = self.data_stamp[s_begin:s_end]
        # 提取输出序列的时间特征
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # 返回输入序列、输出序列及其时间特征
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask

