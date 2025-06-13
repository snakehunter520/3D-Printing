from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, GPT2LMHeadModel, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()
device = "cuda"

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2).to(device)
        self.linear = nn.Linear(nf, target_window).to(device)
        self.dropout = nn.Dropout(head_dropout).to(device)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        # 初始化LLM模型（保持原代码结构）
        if configs.llm_model == 'LLAMA':
            self.llama_config = LlamaConfig.from_pretrained('/data/llm-models/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    '/data/llm-models/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                )
            except EnvironmentError: # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    '/data/llm-models/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    '/data/llm-models/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    '/data/llm-models/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained(r'E:\WuX\时间序列-2\时间序列\gpt2')
            print("模型：GPT2")
            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model =  GPT2LMHeadModel.from_pretrained(
                    r'E:\WuX\时间序列-2\时间序列\gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    r'E:\WuX\时间序列-2\时间序列\gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    r'E:\WuX\时间序列-2\时间序列\gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    r'E:\WuX\时间序列-2\时间序列\gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained(r'E:\WuX\时间序列-2\时间序列\bert-base-uncased')
            print("==============")
            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    r'E:\WuX\时间序列-2\时间序列\bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    r'E:\WuX\时间序列-2\时间序列\bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )
                
            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    r'E:\WuX\时间序列-2\时间序列\bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    '/../../../bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')
        # 将模型加载到GPU上
        self.llm_model.to(device)
        print("llm_model.device",self.llm_model.device)
        # 将分词器加载到GPU上
        # self.tokenizer.to(device)
        # print("tokenizer",self.tokenizer.device)
        self.d_model = configs.d_model

        self.projection = nn.Linear(self.d_model, self.llm_model.config.hidden_size)
        print(f"[DEBUG] 投影层维度: 输入{self.d_model} -> 输出{self.llm_model.config.hidden_size}")  # 应为768
        
        
        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 1)
        print(f"[DEBUG] 分块数量: seq_len={configs.seq_len}, patch_len={self.patch_len}, stride={self.stride} -> {self.patch_nums} patches")
        
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = """ The time sequence data of 3D printing process is an important data for monitoring the printing process. This data monitors the triaxial acceleration of XYZ axis as well as the temperature of the nozzle, the temperature of the hot bed and the room temperature in the 3D printing process. The data collection frequency is 30HZ. Each data point consists of a target value "print status" and 11 running states, where a target value of "0" means normal printing and a target value of "1" means abnormal printing."""

        self.dropout = nn.Dropout(configs.dropout).to(device)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout).to(device)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens).to(device)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm).to(device)

        # self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        # self.head_nf = self.d_ff * self.patch_nums
        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums
        
        self.anomaly_classifier = nn.Sequential(
            nn.Linear(self.llm_model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(256, 4)
        ).to(device)
        print(f"[DEBUG] 分类器输入维度: {self.llm_model.config.hidden_size}")  # 应为768
        # if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
        #     self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
        #                                          head_dropout=configs.dropout)
        # else:
        #     raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'anomaly_classification':
            return self.classify(x_enc, x_mark_enc)
        elif self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None


    def classify(self, x_enc, x_mark_enc):
        B, T, N = x_enc.size()
        TOTAL_LAYERS = 206
        TEMP_MIN = 0
        TEMP_MAX = 300

        # 特征索引定义
        TIME_STAMP_IDX = 0
        ACC_X_IDX = 1
        NOZZLE_TEMP_IDX = 4
        BED_TEMP_IDX = 5
        CURRENT_LAYER_IDX = 7
        TOTAL_LAYERS_IDX = 8

        # 特征提取
        nozzle_temp = x_enc[:, :, NOZZLE_TEMP_IDX]
        bed_temp = x_enc[:, :, BED_TEMP_IDX]
        accel = x_enc[:, :, ACC_X_IDX:ACC_X_IDX+3]

        # 层数计算（修改部分）
        current_layer = x_enc[:, -1, CURRENT_LAYER_IDX].int()
        total_layers = x_enc[:, -1, TOTAL_LAYERS_IDX].int()

        # 动态统计量
        temp_grad = nozzle_temp.diff(dim=1).mean(dim=1)
        accel_variance = accel.var(dim=1).mean(dim=1)
        
        
        
        # 构建prompt（修改部分）
        prompt = []
        for b in range(B):
            nozzle_actual = nozzle_temp[b,-1] * (TEMP_MAX - TEMP_MIN) + TEMP_MIN
            bed_actual = bed_temp[b,-1] * (TEMP_MAX - TEMP_MIN) + TEMP_MIN
            
            layers = x_enc[b, :, CURRENT_LAYER_IDX].int()
            
            print(f"Batch {b} layers:", layers)
            prompt_template = (
                f"<|start_prompt|>3D printing status analysis:{self.description}\n"
                f"- Nozzle temperature: {nozzle_actual:.1f}°C (Δ {temp_grad[b].item():.2f}°C/s)\n"
                f"- Bed temperature: {bed_actual:.1f}°C\n"
                f"- Acceleration variance: {accel_variance[b].item():.4f}\n"
                f"- Current layer: {current_layer[b].item()}\n"  # 修改这里
                f"Task: Predict anomaly probability in next 5 seconds from the following types:\n"
                f"1. Layer adhesion failure 2. Nozzle clogging 3. Thermal runaway 4. Mechanical vibration<|<end_prompt>|>"
            )
            print(prompt_template)
            prompt.append(prompt_template)
            
        print("==================", prompt)
        
        # 数据归一化
        x_enc = self.normalize_layers(x_enc, 'norm')

        # 维度调整 (B, N, T)
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        B_orig, N, T = x_enc.size()

        # 合并batch和特征维度（保持连续性）
        x_enc = x_enc.view(-1, 1, T)  # (B*N, 1, T)
        # print(f"合并后输入形状: {x_enc.shape}")

        # 验证patch参数
        assert T >= self.patch_len, (
            f"时间序列长度{T}必须 ≥ patch长度{self.patch_len}"
        )

        # 时序特征提取
        enc_out, _ = self.patch_embedding(x_enc.to(torch.bfloat16))
        # print(f"PatchEmbedding输出形状: {enc_out.shape}")

        # 直接从输出张量获取实际patch数量
        actual_num_patches = enc_out.size(1)
        # print(f"实际patch数量: {actual_num_patches}")

        # 计算总元素数验证
        total_elements = B_orig * N * actual_num_patches * enc_out.size(-1)
        assert enc_out.numel() == total_elements, (
            f"元素数不匹配: 输入{enc_out.numel()} vs 预期{total_elements}"
        )
        
        # # 计算实际patch数量
        # num_patches = (T - self.patch_len) // self.stride + 1
        # print(f"理论patch数量: {num_patches}")

        # 连续化处理+维度重塑
        enc_out = enc_out.contiguous().reshape(
            B_orig,  # 原始batch size
            N * actual_num_patches,  # 总patch数 (特征数 × 每个特征的patch数)
            -1  # 自动推断d_model维度
        )
        # print(f"重塑后维度: {enc_out.shape}")

        # 提示词编码
        prompt_ids = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=1024
        ).input_ids.to(x_enc.device)

        # 获取LLM embeddings
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_ids)
        # print(f"Prompt嵌入形状: {prompt_embeddings.shape}")

        # 特征投影对齐
        if enc_out.size(-1) != self.llm_model.config.hidden_size:
            enc_out = self.projection(enc_out)
            # print(f"投影后维度: {enc_out.shape}")

        # 多模态融合
        combined_embeds = torch.cat([
            prompt_embeddings, 
            enc_out
        ], dim=1).contiguous()
        # print(f"融合后总维度: {combined_embeds.shape}")

        # LLM处理
        llm_output = self.llm_model(inputs_embeds=combined_embeds).last_hidden_state

        # 提取时序特征
        enc_output = llm_output[:, prompt_embeddings.size(1):, :self.d_ff]
        # print(f"特征输出形状: {enc_output.shape}")

        # 分类器输入适配
        enc_output = enc_output.view(
            B_orig, 
            N, 
            actual_num_patches,  # 每个特征的patch数
            self.d_ff
        ).permute(0, 3, 1, 2)  # (B, N, d_ff, num_patches)
        # print(f"分类器输入形状: {enc_output.shape}")
        
        enc_output = enc_output.reshape(B_orig, self.d_ff, N * actual_num_patches)
        # print(f"调整后的分类器输入形状: {enc_output.shape}")
        
        # 异常分类
        logits = self.anomaly_classifier(enc_output)
        return torch.softmax(logits, dim=-1)
    
    # def classify(self, x_enc, x_mark_enc):
    #     # 维度处理 (B: batch_size, T: 序列长度, N: 特征数)
    #     B, T, N = x_enc.size()
    #     TOTAL_LAYERS = 206
    #     TEMP_MIN = 0
    #     TEMP_MAX = 300

    #     # 特征索引定义
    #     TIME_STAMP_IDX = 0
    #     ACC_X_IDX = 1
    #     NOZZLE_TEMP_IDX = 4
    #     BED_TEMP_IDX = 5
    #     CURRENT_LAYER_IDX = 7
    #     TOTAL_LAYERS_IDX = 8

    #     # 特征提取 ---------------------------------------------------------
    #     nozzle_temp = x_enc[:, :, NOZZLE_TEMP_IDX]  # 喷嘴温度序列
    #     bed_temp = x_enc[:, :, BED_TEMP_IDX]        # 热床温度序列
    #     accel = x_enc[:, :, ACC_X_IDX:ACC_X_IDX+3]  # 三轴加速度（X/Y/Z）

    #     # 动态统计量计算
    #     temp_grad = nozzle_temp.diff(dim=1).mean(dim=1)    # 温度变化率
    #     accel_variance = accel.var(dim=1).mean(dim=1)      # 加速度方差

    #     # 构建提示词(prompt) -----------------------------------------------
    #     prompt = []
    #     for b in range(B):
    #         # 反归一化得到实际物理值
    #         nozzle_actual = nozzle_temp[b,-1] * (TEMP_MAX - TEMP_MIN) + TEMP_MIN
    #         bed_actual = bed_temp[b,-1] * (TEMP_MAX - TEMP_MIN) + TEMP_MIN
            
    #         # 获取当前层数（取序列最后一个时间点的值）
    #         current_layer = x_enc[b, -1, CURRENT_LAYER_IDX].int().item()
    #         total_layers = x_enc[b, -1, TOTAL_LAYERS_IDX].int().item()

    #         # 动态生成提示模板
    #         prompt_template = (
    #                 f"<|system|>You are an expert in 3D printing process monitoring. Analyze the following sensor data:\n"
    #                 f"- Nozzle Temp: {nozzle_actual:.1f}°C (Δ {temp_grad[b]:.2f}°C/s)\n"
    #                 f"- Bed Temp: {bed_actual:.1f}°C\n"
    #                 f"- XYZ Acceleration Variance: {accel_variance[b]:.4f}\n"
    #                 f"- Layer Progress: {current_layer}/{total_layers}\n"
    #                 f"<|user|>Diagnose potential anomalies in the next 5 seconds from:\n"
    #                 f"1. Layer Adhesion Failure\n2. Nozzle Clogging\n3. Thermal Runaway\n4. Mechanical Resonance\n"
    #                 f"Provide a technical explanation and confidence level for each risk.<|assistant|>"
    #             )
    #         prompt.append(prompt_template)

    #     # 数据预处理 -------------------------------------------------------
    #     # 归一化处理
    #     x_enc = self.normalize_layers(x_enc, 'norm')  # (B, T, N)

    #     # 维度调整 (B, N, T)
    #     x_enc = x_enc.permute(0, 2, 1).contiguous()
    #     B_orig, N, T = x_enc.size()

    #     # 合并batch和特征维度
    #     x_enc = x_enc.view(-1, 1, T)  # (B*N, 1, T)

    #     # 时序特征提取 -----------------------------------------------------
    #     enc_out, _ = self.patch_embedding(x_enc.to(torch.bfloat16))  # (B*N, num_patches, d_model)
    #     actual_num_patches = enc_out.size(1)  # 实际patch数量

    #     # 维度重塑
    #     enc_out = enc_out.contiguous().reshape(
    #         B_orig, 
    #         N * actual_num_patches, 
    #         -1
    #     )  # (B_orig, N*num_patches, d_model)

    #     # 提示词编码 -------------------------------------------------------
    #     prompt_embeddings = self.llm_model.get_input_embeddings()(
    #         self.tokenizer(
    #             prompt, 
    #             return_tensors="pt", 
    #             padding=True, 
    #             truncation=True, 
    #             max_length=1024,
    #             add_special_tokens=True
    #         ).input_ids.to(x_enc.device)
    #     )  # (B, prompt_len, hidden_size)

    #     # 特征对齐投影
    #     if enc_out.size(-1) != self.llm_model.config.hidden_size:
    #         enc_out = self.projection(enc_out)  # (B_orig, N*num_patches, hidden_size)

    #     # 多模态融合 ------------------------------------------------------
    #     combined_embeds = torch.cat([
    #         prompt_embeddings, 
    #         enc_out
    #     ], dim=1).contiguous()  # (B, prompt_len + N*num_patches, hidden_size)

    #     # LLM推理 --------------------------------------------------------
    #     llm_outputs = self.llm_model(
    #         inputs_embeds=combined_embeds,
    #         return_dict=True
    #     )
    #     last_hidden_states = llm_outputs[0]  # (B, total_seq_len, hidden_size)
    #     print("============")
    #     # 文本生成解释 ----------------------------------------------------
    #     generated_ids = self.llm_model.generate(
    #         inputs_embeds=combined_embeds,
    #         max_new_tokens=150,
    #         num_beams=5,
    #         early_stopping=True,
    #         temperature=0.9,
    #         top_p=0.95,
    #         pad_token_id=self.tokenizer.eos_token_id,
    #         repetition_penalty=1.1
    #     )
    #     print("============")
    #     generated_text = self.tokenizer.batch_decode(
    #         generated_ids, 
    #         skip_special_tokens=True, 
    #         clean_up_tokenization_spaces=True
    #     )
    #     print(generated_text)
    #     # 分类特征提取 ----------------------------------------------------
    #     enc_output = last_hidden_states[:, prompt_embeddings.size(1):, :self.d_ff]  # (B, N*num_patches, d_ff)
        
    #     # 维度调整
    #     enc_output = enc_output.view(
    #         B_orig, 
    #         N, 
    #         actual_num_patches,
    #         self.d_ff
    #     ).permute(0, 3, 1, 2)  # (B, d_ff, N, num_patches)

    #     # 分类头处理 -----------------------------------------------------
    #     enc_output = enc_output.reshape(B_orig, self.d_ff, N * actual_num_patches)
    #     logits = self.anomaly_classifier(enc_output.mean(dim=-1))  # (B, 4)

        
    #     return {
    #         'prediction': torch.softmax(logits, dim=-1),  # 分类概率
    #         'explanation': generated_text                 # 生成解释
    #     }    

    # def classify(self, x_enc, x_mark_enc):
    #     B, T, N = x_enc.size()
    #     TOTAL_LAYERS = 206
    #     TEMP_MIN = 0
    #     TEMP_MAX = 300

    #     # ========== 特征提取 ==========
    #     TIME_STAMP_IDX = 0
    #     ACC_X_IDX = 1
    #     NOZZLE_TEMP_IDX = 4
    #     BED_TEMP_IDX = 5
    #     CURRENT_LAYER_IDX = 7
    #     TOTAL_LAYERS_IDX = 8

    #     # 特征工程
    #     nozzle_temp = x_enc[:, :, NOZZLE_TEMP_IDX]  # (B, T)
    #     bed_temp = x_enc[:, :, BED_TEMP_IDX]       # (B, T)
    #     accel = x_enc[:, :, ACC_X_IDX:ACC_X_IDX+3] # (B, T, 3)

    #     # ========== 动态统计量 ==========
    #     temp_grad = nozzle_temp.diff(dim=1).mean(dim=1)  # (B,)
    #     accel_variance = accel.var(dim=1).mean(dim=1)    # (B,)

    #     # ========== 数据预处理 ==========
    #     x_enc = self.normalize_layers(x_enc, 'norm')  # (B, T, N)
    #     x_enc = x_enc.permute(0, 2, 1)  # (B, N, T)

    #     # ========== 分块嵌入核心修改 ==========
    #     enc_out_list = []
    #     for i in range(N):
    #         # 单个特征分块处理 (B, 1, T)
    #         x_feature = x_enc[:, i:i+1, :]
            
    #         # 计算实际分块数量
    #         seq_len = x_feature.size(-1)
    #         num_patches = (seq_len - self.patch_len) // self.stride + 1
    #         print(f"[DEBUG] 特征{i}分块数: {num_patches} (seq_len={seq_len})")

    #         # 分块嵌入
    #         enc_feature, _ = self.patch_embedding(x_feature)  # (B, num_patches, d_model)
    #         enc_out_list.append(enc_feature)

    #     # 合并所有特征的分块结果 (B, N*num_patches, d_model)
    #     enc_out = torch.cat(enc_out_list, dim=1)
    #     print(f"[DEBUG] 合并后维度: {enc_out.shape}")

    #     # ========== 维度对齐关键步骤 ==========
    #     # 展平并调整维度到d_model
    #     flattened_dim = enc_out.size(1) * enc_out.size(2)  # N*num_patches*d_model
    #     print(f"[DEBUG] 展平前维度: {enc_out.shape} -> 展平后维度: ({B}, {flattened_dim})")
        
    #     # 动态维度调整层
    #     if not hasattr(self, 'dim_adjust'):
    #         self.dim_adjust = nn.Linear(flattened_dim, self.d_model).to(device)
    #         print(f"[INIT] 创建动态调整层: {flattened_dim} -> {self.d_model}")
        
    #     enc_out = enc_out.view(B, -1)  # (B, N*num_patches*d_model)
    #     enc_out = self.dim_adjust(enc_out)  # (B, d_model)
    #     print(f"[DEBUG] 调整后维度: {enc_out.shape}")

    #     # ========== 投影层验证 ==========
    #     print(f"[VALID] 投影层输入维度: {enc_out.shape[-1]} vs 权重维度: {self.projection.in_features}")
    #     enc_out = self.projection(enc_out)  # (B, llm_hidden_size)
    #     print(f"[DEBUG] 投影后维度: {enc_out.shape}")

    #     # ========== 提示工程增强 ==========
    #     prompt = []
    #     for b in range(B):
    #         # 实际温度值反归一化
    #         nozzle_actual = nozzle_temp[b,-1] * (TEMP_MAX - TEMP_MIN) + TEMP_MIN
    #         bed_actual = bed_temp[b,-1] * (TEMP_MAX - TEMP_MIN) + TEMP_MIN
            
    #         # 动态构建提示语
    #         prompt_template = (
    #             f"<|start_prompt|>3D打印状态分析：{self.description}\n"
    #             f"- 喷嘴温度：{nozzle_actual:.1f}°C (梯度 {temp_grad[b].item():.2f}°C/s)\n"
    #             f"- 热床温度：{bed_actual:.1f}°C\n"
    #             f"- 加速度方差：{accel_variance[b].item():.4f}\n"
    #             f"- 当前层数：{int(x_enc[b,CURRENT_LAYER_IDX,-1]*TOTAL_LAYERS)}/{TOTAL_LAYERS}\n"
    #             "请预测未来5秒内可能发生的异常类型概率：\n"
    #             "1.层间粘接失效 2.喷嘴堵塞 3.热失控 4.机械振动<|<end_prompt>|>"
    #         )
    #         prompt.append(prompt_template)

    #     # ========== 多模态融合 ==========
    #     # 生成提示词嵌入
    #     prompt_ids = self.tokenizer(
    #         prompt, 
    #         return_tensors="pt", 
    #         padding=True, 
    #         truncation=True, 
    #         max_length=512
    #     ).input_ids.to(device)
        
    #     # 获取LLM嵌入
    #     prompt_embeds = self.llm_model.get_input_embeddings()(prompt_ids)  # (B, seq_len, hidden_size)
        
    #     # 拼接时序特征 (在序列末尾添加)
    #     combined_embeds = torch.cat([
    #         prompt_embeds, 
    #         enc_out.unsqueeze(1)  # (B, 1, hidden_size)
    #     ], dim=1)  # (B, seq_len+1, hidden_size)
    #     print(f"[DEBUG] 融合后总维度: {combined_embeds.shape}")

    #     # ========== LLM处理 ==========
    #     llm_output = self.llm_model(inputs_embeds=combined_embeds).last_hidden_state  # (B, seq_len+1, hidden_size)
        
    #     # ========== 分类器增强 ==========
    #     # 提取时序特征对应的输出
    #     time_feature = llm_output[:, -1, :]  # 取最后一个位置的表示 (B, hidden_size)
    #     print(f"[DEBUG] 分类器输入维度: {time_feature.shape}")
        
    #     logits = self.anomaly_classifier(time_feature)
        
    #     # 通过分类器层
    #     if self.training:
    #         # 训练时进行维度扩展匹配
    #         # 假设每个样本需要预测pred_len个时间点
    #         logits = logits.unsqueeze(1).repeat(1, self.pred_len, 1)  # (B, pred_len, 4)
    #         print(f"[TRAIN] 训练模式输出维度: {logits.shape}")
    #         return logits
    #     else:
    #         # 验证/测试时保持单点预测
    #         print(f"[EVAL] 验证模式输出维度: {logits.shape}")
    #         return torch.softmax(logits, dim=-1)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        
        
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids

        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
