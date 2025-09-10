import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from collections import OrderedDict

def get_device(force_cpu=False, device_config=None):
    """检测并返回可用的设备，支持多GPU"""
    if force_cpu:
        device = torch.device("cpu")
        print("强制使用CPU进行计算")
        return device, False, []
    
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA不可用，使用CPU进行计算")
        return device, False, []
    
    # 获取可用的GPU数量和信息
    gpu_count = torch.cuda.device_count()
    print(f"🎮 检测到 {gpu_count} 个可用GPU:")
    
    available_gpus = []
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        available_gpus.append(i)
    
    # 检查是否启用多GPU训练
    use_multi_gpu = gpu_count > 1
    if device_config and 'multi_gpu' in device_config:
        use_multi_gpu = device_config['multi_gpu'] and gpu_count > 1
    
    # 选择主GPU设备
    primary_device = torch.device("cuda:0")
    
    if use_multi_gpu:
        print(f"🚀 启用多GPU训练，使用 {gpu_count} 个GPU")
        return primary_device, True, available_gpus
    else:
        print(f"📱 使用单GPU训练: {torch.cuda.get_device_name(0)}")
        return primary_device, False, [0]



class RMSNorm(nn.Module):
    def __init__(self, layer_shape, eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()

        # torch中register_parameter()功能为：向我们建立的网络module添加parameter
        # 因此，我们需要对pytorch官方封装好的RMSNorm功能模块添加一个可以训练参数的层，命名为scale，并初始化为形状为layer_shape，所有值为1的张量矩阵。
        self.register_parameter("scale", nn.Parameter(torch.ones(layer_shape)))

    def forward(self, x):
        # 计算Frobenius范数（球某个矩阵中所有元素的平方和再开方得到，该范数用来衡量矩阵的大小，详情请百度）, RMS = 1/sqrt(N) * Frobenius
        # 具体来说，torch.linalg.norm(x, dim=(1, 2))计算了x在第1和第2维度上的范数。然后，将结果乘以x[0].numel() ** -.5。x[0].numel()表示x第一个元素（即x的第一行）的元素个数，** -.5表示求平方根的倒数。
        ff_rms = torch.linalg.norm(x, dim=(1,2)) * x[0].numel() ** -.5
        # print(ff_rms.shape)
        # 将ff_rms算子应用于输入的张量x，依据公式，做除法，因为输入向量x是三维的，因此需要对ff_rms进行升两维，也变成三维的张量。这样可以进行元素之间的计算。
        raw = x / ff_rms.unsqueeze(-1).unsqueeze(-1)
        # print(raw.shape)
        # 返回scale缩放后归一化的张量
        # print(self.scale[:x.shape[1], :].unsqueeze(0) * raw)
        return self.scale[:x.shape[1], :].unsqueeze(0) * raw

def get_rotary_matrix(context_window, embedding_dim, device=None):
    # 初始化一个0填充，形状为（context_window, embedding_dim, embedding_dim）的张量矩阵，其中context_window为token数量，后面两个embedding_dim组成正方形矩阵，与后面的attention计算对齐格式
    R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False, device=device)

    # 遍历每一个位置的token
    for position in range(context_window):
        # 还记得我的上一篇文章中说的，对于特征，两两组合吗，因此需要循环的次数为embedding_dim除以2
        for i in range(embedding_dim // 2):
            # 设置θ值，采样频率，或者说旋转频率，旋转角都可以，除以embedding_dim防止梯度问题。
            theta = 10000. ** (-2. * (i - 1) / embedding_dim)
            # 根据欧拉公式，计算旋转的角度，分别有sin 和cos，将计算拉到复数空间，并将旋转角度应用在上面的0填充的矩阵
            m_theta = position * theta
            R[position, 2 * i, 2 * i] = np.cos(m_theta)
            R[position, 2 * i, 2 * i + 1] = -np.sin(m_theta)
            R[position, 2 * i + 1, 2 * i] = np.sin(m_theta)
            R[position, 2 * i + 1, 2 * i + 1] = np.cos(m_theta)
            # 得到的结果是旋转位置编码矩阵，到这里还没覆盖到attention
    return R

# 此为单头注意力机制
class RoPEMaskedAttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 计算Q权重矩阵
        self.w_q = nn.Linear(config['d_model'], config['d_model'], bias=False)
        # 计算K权重矩阵
        self.w_k = nn.Linear(config['d_model'], config['d_model'], bias=False)
        # 计算V权重矩阵
        self.w_v = nn.Linear(config['d_model'], config['d_model'], bias=False)
        
        self.context_window = config['context_window']
        self.embedding_dim = config['d_model']
        
        # 预计算并缓存旋转位置编码矩阵 - 不作为模型参数保存
        self.R_cache = None
        self.cache_device = None

    def _get_rotary_matrix(self, device):
        """获取旋转位置编码矩阵，使用缓存避免重复计算"""
        # 如果缓存不存在或设备不匹配，重新创建
        if self.R_cache is None or self.cache_device != device:
            R = torch.zeros((self.context_window, self.embedding_dim, self.embedding_dim), 
                          requires_grad=False, device=device)
            
            # 遍历每一个位置的token
            for position in range(self.context_window):
                # 计算旋转角度
                for i in range(self.embedding_dim // 2):
                    theta = 10000. ** (-2. * (i - 1) / self.embedding_dim)
                    m_theta = position * theta
                    R[position, 2 * i, 2 * i] = np.cos(m_theta)
                    R[position, 2 * i, 2 * i + 1] = -np.sin(m_theta)
                    R[position, 2 * i + 1, 2 * i] = np.sin(m_theta)
                    R[position, 2 * i + 1, 2 * i + 1] = np.cos(m_theta)
            
            # 缓存矩阵和设备信息
            self.R_cache = R
            self.cache_device = device
            
        return self.R_cache

    def forward(self, x, return_attn_weights=False):
        # 前向传播时，输入矩阵的形状为(batch, sequence length, dimension)
        b, m, d = x.shape  # batch size, sequence length, dimension

        # 线性变换Q,K,V
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # 获取缓存的旋转位置编码矩阵
        R = self._get_rotary_matrix(x.device)
        
        # 将旋转位置编码应用于Q和K，只使用需要的部分
        q_rotated = (torch.bmm(q.transpose(0, 1), R[:m])).transpose(0, 1)
        k_rotated = (torch.bmm(k.transpose(0, 1), R[:m])).transpose(0, 1)

        # 对注意力机制点积进行等比例缩放
        activations = F.scaled_dot_product_attention(
            q_rotated, k_rotated, v, dropout_p=0.1, is_causal=True
        )
        
        # 如果需要返回注意力权重
        if return_attn_weights:
            # 生成上三角布尔掩码（未来位置为True）
            attn_mask = torch.triu(torch.ones(m, m, device=q_rotated.device), diagonal=1).bool()
            # 计算原始注意力分数
            attn_scores = torch.bmm(q_rotated, k_rotated.transpose(1,2)) / np.sqrt(d)
            # 遮盖未来位置（设为-1e9）
            attn_scores = attn_scores.masked_fill(attn_mask, -1e9)
            # Softmax归一化
            attn_weights = F.softmax(attn_scores, dim=-1)
            return activations, attn_weights

        return activations

# 单头注意力机制实现完毕，下面实现多头注意力机制
class RoPEMaskedMultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 一个注意力机制头对象构建完毕了，多头的，首先多次创建这个对象。生成多个注意力机制头，塞到一个列表里。
        self.heads = nn.ModuleList([
            RoPEMaskedAttentionHead(config) for _ in range(config['n_heads'])
        ])
        # 在模型结构上，创建一个线性层（隐藏层），用于线型输出注意力机制头输出的张量矩阵，寻找多头之间的特征，但是更主要的是，x经过多头计算后形状改变了，创建线性层，让张量矩阵变回原来输入的形状。
        # 同时为了防止过拟合，使用随机神经元失活，比率0.1
        # 线性层输入形状：注意力机制的头数，乘以矩阵的维度，关联到俺的上一篇文章，就是key矩阵，在多头之间共享权重，减少计算的思维。 输出为：模型的embedding维度数
        self.linear = nn.Linear(config['n_heads'] * config['d_model'], config['d_model'])
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # 输入矩阵形状x： (batch, sequence length, dimension)

        # 每一个注意力机制头，都传入X进行计算。（这个地方开启并行执行会不会快一些，但是不知道pytorch是不是自动调用并行）
        heads = [h(x) for h in self.heads]  # 不返回注意力权重，只返回激活值
        # 输入张量x经过多个头计算attention（同时，attention是已经覆盖了RoPE的），重新拼接成新的矩阵，重新放入变量x。到这里你应该觉得：那矩阵形状不就变了吗
        x = torch.cat(heads, dim=-1)

        # 这不，线性层的作用来了
        x = self.linear(x)

        # 随机失活一下，防止过拟合
        x = self.dropout(x)
        return x





class SwiGLU(nn.Module):
    """SwiGLU激活函数 - 改进版本"""
    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
            
        # 门控线性层
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        # 上投影层
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        # 下投影层
        self.down_proj = nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, x):
        # SwiGLU: swish(gate_proj(x)) * up_proj(x)
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # Swish激活函数: x * sigmoid(x)
        swish_gate = gate * torch.sigmoid(gate)
        # 门控机制
        hidden = swish_gate * up
        # 下投影
        output = self.down_proj(hidden)
        return output


# 现在我们拥有了所有的算子，RMS，ROPE,SWIGLU，我们搭建我们的Llmpvq！ 首先实现Llmpvq的功能块，然后堆叠。
# 功能没什么好讲的，如果仔细看到了这里，下面的每一行代码都难不住你。
class LlmpvqBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rms_1 = RMSNorm((config['context_window'], config['d_model']))
        self.attention = RoPEMaskedMultiheadAttention(config)
        self.rms_2 = RMSNorm((config['context_window'], config['d_model']))
        
        # 改进的FFN设计，使用SwiGLU激活函数
        ffn_dim = config.get('ffn_dim', config['d_model'] * 4)
        self.feedforward = nn.Sequential(
            SwiGLU(config['d_model'], ffn_dim),
            nn.Dropout(config.get('dropout', 0.1))
        )

    def forward(self, x):
        # Pre-norm架构：先归一化再计算
        # 注意力机制的残差连接
        x = x + self.attention(self.rms_1(x))
        
        # FFN的残差连接
        x = x + self.feedforward(self.rms_2(x))
        
        return x

# 现在，我们组装Llmpvq - 优化版本
class Llmpvq(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token嵌入层
        self.embeddings = nn.Embedding(config['vocab_size'], config['d_model'])
        
        # 位置嵌入（可选，因为我们使用RoPE）
        # self.pos_embeddings = nn.Embedding(config['context_window'], config['d_model'])
        
        # Dropout层
        self.dropout = nn.Dropout(config.get('dropout', 0.1))
        
        # Transformer块堆叠
        self.Llmpvq_blocks = nn.Sequential(
            OrderedDict([(f"Llmpvq_{i}", LlmpvqBlock(config)) for i in range(config['n_layers'])])
        )
        
        # 最终的层归一化
        self.final_norm = RMSNorm((config['context_window'], config['d_model']))
        
        # 输出投影层（语言模型头）
        self.lm_head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        
        # 权重初始化
        self.apply(self._init_weights)
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"模型总参数量: {total_params/1e9:.2f}B ({total_params:,})")
        print(f"可训练参数量: {trainable_params/1e9:.2f}B ({trainable_params:,})")
        print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")

    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.shape
        
        # Token嵌入
        x = self.embeddings(idx)  # (batch_size, seq_len, d_model)
        
        # 添加dropout
        x = self.dropout(x)
        
        # 通过Transformer块
        x = self.Llmpvq_blocks(x)
        
        # 最终层归一化
        x = self.final_norm(x)
        
        # 语言模型头
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)

        # 计算损失
        if targets is not None:
            # 重塑张量以计算交叉熵损失
            logits_flat = logits.view(-1, self.config['vocab_size'])
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-1, reduction='mean')
            # 确保损失是标量 - 在多GPU环境下特别重要
            if loss.dim() > 0:
                loss = loss.mean()
            return logits, loss
        else:
            return logits

# 便利函数
def create_model(config=None):
    """创建模型实例，支持多GPU"""
    if config is None:
        raise ValueError("必须提供配置参数,config不能为None")
    
    device = config.get('device', torch.device('cpu'))
    multi_gpu = config.get('multi_gpu', False)
    gpu_ids = config.get('gpu_ids', [])
    
    # 如果使用GPU，进行优化设置
    if device.type == 'cuda':
        print("🚀 配置GPU优化...")
        
        # 启用TensorFloat-32 (TF32) 加速 (A100/RTX30系列及以上)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # 启用融合内核优化
        torch.backends.cuda.enable_flash_sdp(True)  # Flash Attention
        
        # 预分配内存避免碎片化
        print("预分配GPU内存...")
        temp_tensor = torch.randn(1000, 1000, device=device)
        del temp_tensor
        torch.cuda.empty_cache()
        print("GPU优化配置完成")
    
    print("🏗️ 创建模型...")
    model = Llmpvq(config)
    
    # 将模型移动到主设备
    model = model.to(device)
    print(f"模型已移动到主设备: {device}")
    
    # 如果启用多GPU，使用DataParallel
    if multi_gpu and len(gpu_ids) > 1:
        print(f"🎯 启用多GPU并行训练，使用GPU: {gpu_ids}")
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"✅ DataParallel已启用，主GPU: {device}")
        
        # 计算多GPU下的有效批次大小
        effective_batch_size = config.get('batch_size', 32) * len(gpu_ids)
        print(f"📊 有效批次大小: {config.get('batch_size', 32)} × {len(gpu_ids)} = {effective_batch_size}")
    
    # 确保所有GPU操作完成
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    return model

# 优化器、调度器和训练环境初始化函数已移动到 train.py