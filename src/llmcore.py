import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd
import urllib.request
import os
from collections import OrderedDict

def get_device(force_cpu=False):
    """检测并返回可用的设备"""
    if force_cpu:
        device = torch.device("cpu")
        print("强制使用CPU进行计算")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用GPU: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("CUDA不可用，使用CPU进行计算")
    return device

def get_default_config(force_cpu=False):
    """获取默认配置"""
    device = get_device(force_cpu=force_cpu)
    return {
        'batch_size': 32, # 一次batch会有多少个随机点
        'context_window': 16, # 滑动窗口采样，设置采样大小
        'vocab_size': 4325, # 咱们的西游记数据集，一共包含4325个不重复的汉字，标点符号
        'd_model': 128, #模型为128维的embedding
        'epochs': 5000, # 训练轮次 - 减少到5000轮，配合早停机制
        'log_interval': 10, # 每10个batch打印一次log
        'n_heads': 8, # 32个注意力机制头，我们来8个吧
        'n_layers': 4, # 根据传入的堆叠层数，创建Llama功能块，注意OrderedDict为一种特殊类型的字典数据，保留字典写入的顺序，先插入的数据在前，后插入的数据在后。
        # 这里，我们将llama的功能块堆叠4层
        'device': device, # 计算设备
    }

def download_and_prepare_data(data_file="xiyouji.txt", force_download=False):
    """下载并准备数据集"""
    if not os.path.exists(data_file) or force_download:
        print(f"正在下载数据集到 {data_file}...")
        url = "https://raw.githubusercontent.com/mc112611/PI-ka-pi/main/xiyouji.txt"
        urllib.request.urlretrieve(url, data_file)
        print("数据集下载完成")
    
    # 读取数据
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.read()
    
    # 创建词表
    vocab = sorted(list(set(lines)))
    print(f'词表大小: {len(vocab)}')
    
    # 创建编码映射
    itos = {i: ch for i, ch in enumerate(vocab)}
    stoi = {ch: i for i, ch in enumerate(vocab)}
    
    # 编码整个数据集
    dataset = torch.tensor([stoi[ch] for ch in lines], dtype=torch.int16)
    print(f'数据集形状: {dataset.shape}')
    
    return dataset, vocab, itos, stoi

def encode_text(text, stoi):
    """编码文本为数字序列"""
    return [stoi[ch] for ch in text]

def decode_text(indices, itos):
    """解码数字序列为文本"""
    return ''.join([itos[i] for i in indices])

# 构建batch
def get_batches(data, split, batch_size, context_window, device=None):
    # 切分训练集，验证集，测试集，比例为，训练80%，验证10%，测试10%
    train = data[:int(0.8 * len(data))]
    val = data[int(0.8 * len(data)): int(0.9 * len(data))]
    test = data[int(0.9 * len(data)):]

    # 将全部的训练数据作为batch，验证集，测试集也换个变量存储（单纯为了方便看）
    batch_data = train
    if split == 'val':
        batch_data = val
    if split == 'test':
        batch_data = test

    # 如果使用GPU，先将数据移动到GPU上，避免重复传输
    if device is not None and device.type == 'cuda' and batch_data.device != device:
        batch_data = batch_data.to(device)

    # 在GPU上生成随机索引，避免CPU到GPU的传输
    device_for_randint = device if device is not None else torch.device('cpu')
    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,), device=device_for_randint)

    # 使用更高效的向量化索引方式，避免Python循环
    # 创建索引矩阵，一次性获取所有batch数据
    batch_indices = ix.unsqueeze(1) + torch.arange(context_window, device=ix.device).unsqueeze(0)
    target_indices = batch_indices + 1
    
    x = batch_data[batch_indices].long()
    y = batch_data[target_indices].long()

    # 确保数据在正确的设备上
    if device is not None:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

    # 返回特征值，目标值
    return x, y



# 构造一个评估函数
@torch.no_grad()
def evaluate_loss(model, dataset, config):
    # 评估结果存储变量
    out = {}

    # 将模型置为评估模式
    model.eval()

    # 分别会在训练集和验证集里通过get_batchs()函数取评估数据
    for split in ["train", "val"]:

        losses = []

        # 评估10个batch
        for _ in range(10):
            # 拿到特征值（输入数据），以及目标值（输出数据）
            xb, yb = get_batches(dataset, split, config['batch_size'], config['context_window'], config.get('device'))

            # 把拿到的数据丢进模型，得到loss值
            _, loss = model(xb, yb)

            # 更新loss存储
            losses.append(loss.item())

        # 这里就是大家经常在控制台看到的 "train_loss"  "valid_loss"由来
        out[split] = np.mean(losses)

    # 评估完了，别忘了把模型再置回训练状态，下一个epoch还要继续训练呢
    model.train()

    return out



# 创建一个Adam优化器，基础知识，
# optimizer = torch.optim.Adam(
#     model.parameters(),      # 优化器执行优化全部的模型参数
# )

# 构建训练函数
def train(model, optimizer, dataset, config, scheduler=None, print_logs=False):
    # loss存储
    losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 50  # 早停耐心值

    # 训练时间记录开始时间
    start_time = time.time()

    # 循环训练指定epoch的轮数
    for epoch in range(config['epochs']):
        # 优化器要初始化啊，否则每次训练都是基于上一次训练结果进行优化，效果甚微
        optimizer.zero_grad()

        # 获取训练数据
        xs, ys = get_batches(dataset, 'train', config['batch_size'], config['context_window'], config.get('device'))

        # 前向传播计算概率矩阵与loss
        logits, loss = model(xs, targets=ys)

        # 反向传播更新权重参数，更新学习率优化器
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        # 如果提供学习率调度器，那么学习率会通过调度器进行修改
        if scheduler:
            # 每个batch都更新学习率
            scheduler.step()

        # 打印log
        if epoch % config['log_interval'] == 0:
            # 训练时间
            batch_time = time.time() - start_time

            # 执行评估函数，在训练集和验证集上计算loss
            x = evaluate_loss(model, dataset, config)

            # Store the validation loss
            losses += [x]
            
            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']

            # 打印进度日志
            if print_logs:
                print(f"Epoch {epoch:5d} | train loss {x['train']:.4f} | val loss {x['val']:.4f} | lr {current_lr:.2e} | Time {batch_time:.2f}s | ETA {batch_time * (config['epochs'] - epoch)/config['log_interval']:.1f}s")

            # 将验证loss传递给调度器进行自适应调整
            if scheduler and hasattr(scheduler, 'step'):
                # 对于自定义调度器，传递验证loss
                if hasattr(scheduler, 'plateau_patience'):
                    scheduler.step(val_loss=x['val'])
            
            # 早停机制
            if x['val'] < best_val_loss:
                best_val_loss = x['val']
                patience_counter = 0
                if print_logs:
                    print(f"  ✓ 新的最佳验证loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"  ⚠ 早停触发！验证loss连续{max_patience}轮未改善")
                    break

            # 重置开始时间，用于计算下一轮的训练时间
            start_time = time.time()

    # 上面所有epoch训练结束，打印最终的结果
    print(f"\n=== 训练结束 ===")
    print(f"最佳验证loss: {best_val_loss:.4f}")
    print(f"最终验证loss: {losses[-1]['val']:.4f}")

    # 返还每一步loss值的列表，因为我们要画图，返还的是loss迭代的图像
    return pd.DataFrame(losses).plot()

# 启动训练
# train(model, optimizer)

# 推理函数
def generate(model, itos, config, max_new_tokens=20):
    # 生成随机数，作为输入数据,5行一列，代表输入5个字符。 这个地方可以自行替换其他随机数测试。
    device = config.get('device', torch.device('cpu'))
    idx = torch.randint(1, 4000, (5, 1)).long().to(device)
    print("随机生成的字符:", [itos[i.item()] for i in idx])
    print(idx[:, -config['context_window']:])
    for _ in range(max_new_tokens):
        # 因为推理的时候，依赖后面的n个token，所以滑动窗口要从后往前选择输入数据的倒数几个token，这个是超过字符数量会对输入进行截断，只选取最后几个token：idx[:, -config['context_window']:]
        logits = model(idx[:, -config['context_window']:])
        # print(logits.size())
        # 得到模型输出的结果，进行解码，这里logits[:, -1, :]挺抽象的，实际上第一维度是输入的字符数，第二维度是时间步，第三维度是词表
        # 即，对每一步的解码结果，取最后一个时间步的数据，作为输出的数据。解码的过程是第一次解码，输入5个token，第二次解码依赖的是原来5个token的最后4个，加上上一步解码生成的一个，也是5个token，如此循环。
        last_time_step_logits = logits[:, -1, :]
        # print('last_time_step_logits')
        # print(last_time_step_logits.shape)
        # 计算概率分布
        p = F.softmax(last_time_step_logits, dim=-1)
        # print('p_shape')
        # print(p.shape)
        # 根据概率分布计算下一个token，这里使用 torch.multinomial做的是随机采样
        idx_next = torch.multinomial(p, num_samples=1)
        # print('idx_next_shape')
        # print(idx_next.shape)
        # 将新的idx通过张量拼接写入到解码序列中
        idx = torch.cat([idx, idx_next], dim=-1)
    # 使用之前定义的解码函数，将ID转换为汉字，我们得到的5行21列的数据，来源于每一个输入字符作为开始位置，生成20个字符。 因为5个输入都是0，在词表中编号为0的数据是'\n'。
    print(idx.shape)
    return [decode_text(x, itos) for x in idx.tolist()]

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

# Llama 32个注意力机制头，我们来8个吧


# 我们已经创建完了所需要的算子，  现在积木已创建完毕，将这些积木组合起来！！！！
class RopeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embedding层
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])

        # RMSNorm层
        self.rms = RMSNorm((config['context_window'], config['d_model']))

        # 旋转位置编码器+注意力机制
        self.rope_attention = RoPEMaskedMultiheadAttention(config)

        # 线性层+激活函数变为非线性输出！
        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.ReLU(),
        )

        # 最终的输出，因为需要解码，因为输出的维度与词表大小统一！！！
        self.last_linear = nn.Linear(config['d_model'], config['vocab_size'])

        print("model params:", sum([m.numel() for m in self.parameters()]))
    # 前向传播
    def forward(self, idx, targets=None):
        # embedding，不解释
        x = self.embedding(idx)
        # 归一化数值，不解释
        x = self.rms(x)
        # 相加，解释一下，因为attention是要覆盖到原矩阵的，想象两个形状一样的矩阵为两张纸，左手一张纸，右手一张纸，双手合十，啪！覆盖。 使用加算，就是将两个矩阵中的元素按位置相加！直接覆盖值！
        x = x + self.rope_attention(x)
        # 再归一化！
        x = self.rms(x)
        # 因为直接计算归一化的数值可能出现梯度问题，因此把归一化的值作为修正系数，再覆盖！
        x = x + self.linear(x)
        # 到这里，才是最终输出vocab数量的神经元输出！！！！！！
        logits = self.last_linear(x)

        # 训练阶段有目标值
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss
        # 验证或者推理阶段，目标值y没有！只有结果，没有loss！
        else:
            return logits


class SwiGLU(nn.Module):

    def __init__(self, size):
        super().__init__()
        # 定义一个门控的线性层，输入输出都是门控结构的尺寸
        self.linear_gate = nn.Linear(size, size)
        # 门控结构主干线性层
        self.linear = nn.Linear(size, size)
        # 初始化一个随机数作为beta系数
        self.beta = torch.randn(1, requires_grad=True)

        # nn.Parameter用于指定某一层参数为可学习的，即本来不能通过训练更改参数，现在变成了可以经过训练来更新的参数。
        self.beta = nn.Parameter(torch.ones(1))
        # 将随机数beta指定为一个名为beta的神经网络层
        self.register_parameter("beta", self.beta)

    def forward(self, x):
        # Swish门控但愿的计算：（从括号里开始）对于原始输入的数据张量，经过线性变换乘以beta系数，再经过sigmoid变换为0-1之间的值，再乘以原数据经过门控线性变换。总的来说，线型输出经过非线性变换，再应用到线性变换的结果，元素按位置相乘，修正原本数据张量，就是这个门控结构做的事情。
        swish_gate = self.linear_gate(x) * torch.sigmoid(self.beta * self.linear_gate(x))
        # 将门控结构输出的值再按位乘以线型输出的原数据张量
        # 为啥这么做，我不知道，但是论文复现的代码就是这样滴，有兴趣可以研究一下，我没研究过。
        out = swish_gate * self.linear(x)
        return out


# OK！ 现在我们更新一下，隐藏层维度堆叠多少层，我们先来4层尝尝咸淡！！！！

# 现在我们拥有了所有的算子，RMS，ROPE,SWIGLU，我们搭建我们的LlaMa！ 首先实现LlaMa的功能块，然后堆叠。
# 功能没什么好讲的，如果仔细看到了这里，下面的每一行代码都难不住你。
class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rms = RMSNorm((config['context_window'], config['d_model']))
        self.attention = RoPEMaskedMultiheadAttention(config)
        self.feedforward = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model']),
        )

    def forward(self, x):

        x = self.rms(x)
        x = x + self.attention(x)

        x = self.rms(x)
        x = x + self.feedforward(x)
        return x

# 现在，我们组装LlaMa
class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Embedding不解释
        self.embeddings = nn.Embedding(config['vocab_size'], config['d_model'])
        # 根据传入的堆叠层数，创建Llama功能块，注意OrderedDict为一种特殊类型的字典数据，保留字典写入的顺序，先插入的数据在前，后插入的数据在后。
        # 这里，我们将llama的功能块堆叠4层
        self.llama_blocks = nn.Sequential(
            OrderedDict([(f"llama_{i}", LlamaBlock(config)) for i in range(config['n_layers'])])
        )
        # FFN层，包含：线性层、激活函数非线性变换、再用线性层输出最终解码数值。
        self.ffn = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model']),
            nn.Linear(config['d_model'], config['vocab_size']),
        )

        # 看看咱们的大模型多少参数！
        print("model params:", sum([m.numel() for m in self.parameters()]))

    def forward(self, idx, targets=None):
        # embedding嵌入
        x = self.embeddings(idx)
        # Llama模型计算
        x = self.llama_blocks(x)
        # FFN计算，得到logits
        logits = self.ffn(x)

        # 推理阶段没有目标值，只输出结果
        if targets is None:
            return logits
        # 训练阶段，有目标值，需要输出结果，以及loss，用于反向传播更新权重！
        else:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss

# 便利函数
def create_model(config=None):
    """创建模型实例"""
    if config is None:
        config = get_default_config()
    
    # 如果使用GPU，预分配内存
    if config.get('device', torch.device('cpu')).type == 'cuda':
        print("预分配GPU内存...")
        # 创建一个大的临时张量来预分配内存
        temp_tensor = torch.randn(1000, 1000, device=config['device'])
        del temp_tensor
        torch.cuda.empty_cache()
        print("GPU内存预分配完成")
    
    
    model = Llama(config)
    # 将模型移动到指定设备
    if 'device' in config:
        model = model.to(config['device'])
        print(f"模型已移动到设备: {config['device']}")
    return model

def create_optimizer(model, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8):
    """创建优化器 - 使用更适合大模型的学习率"""
    # return torch.optim.AdamW(  # 使用AdamW，对权重衰减处理更好
    #     model.parameters(),
    #     lr=lr,
    #     betas=betas,
    #     weight_decay=weight_decay,
    #     eps=eps
    # )
    return torch.optim.Adam(model.parameters())

class WarmupCosineScheduler:
    """带预热的余弦退火学习率调度器，包含自适应调整功能"""
    def __init__(self, optimizer, warmup_steps=100, max_steps=10000, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
        
        # 自适应学习率调整
        self.plateau_patience = 10  # loss停滞容忍步数
        self.plateau_factor = 0.5   # loss停滞时的学习率衰减因子
        self.best_loss = float('inf')
        self.plateau_count = 0
        
    def step(self, val_loss=None):
        self.current_step += 1
        
        # 检查是否需要自适应调整学习率
        if val_loss is not None:
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.plateau_count = 0
            else:
                self.plateau_count += 1
                
            # 如果loss停滞，降低学习率
            if self.plateau_count >= self.plateau_patience:
                self.base_lr *= self.plateau_factor
                self.plateau_count = 0
                print(f"  📉 检测到loss停滞，学习率降低至: {self.base_lr:.2e}")
        
        if self.current_step <= self.warmup_steps:
            # 预热阶段：线性增长
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # 余弦退火阶段
            progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        """返回调度器的状态字典"""
        return {
            'current_step': self.current_step,
            'base_lr': self.base_lr,
            'best_loss': self.best_loss,
            'plateau_count': self.plateau_count,
            'warmup_steps': self.warmup_steps,
            'max_steps': self.max_steps,
            'min_lr': self.min_lr,
            'plateau_patience': self.plateau_patience,
            'plateau_factor': self.plateau_factor
        }
    
    def load_state_dict(self, state_dict):
        """加载调度器的状态字典"""
        self.current_step = state_dict['current_step']
        self.base_lr = state_dict['base_lr']
        self.best_loss = state_dict['best_loss']
        self.plateau_count = state_dict['plateau_count']
        self.warmup_steps = state_dict['warmup_steps']
        self.max_steps = state_dict['max_steps']
        self.min_lr = state_dict['min_lr']
        self.plateau_patience = state_dict['plateau_patience']
        self.plateau_factor = state_dict['plateau_factor']

def create_scheduler(optimizer, warmup_steps=100, max_steps=10000, min_lr=1e-6):
    """创建带预热的学习率调度器"""
    return WarmupCosineScheduler(optimizer, warmup_steps, max_steps, min_lr)

def initialize_training_environment(data_file="xiyouji.txt", config=None):
    """初始化训练环境，返回所有必要的组件"""
    if config is None:
        config = get_default_config()
    
    # 准备数据
    dataset, vocab, itos, stoi = download_and_prepare_data(data_file)
    
    # 更新配置中的词表大小
    config['vocab_size'] = len(vocab)
    
    # 如果使用GPU，将数据集移动到GPU上以避免重复传输
    device = config.get('device', torch.device('cpu'))
    if device.type == 'cuda':
        print(f"将数据集移动到GPU: {device}")
        dataset = dataset.to(device)
        # 预热GPU，减少首次运行的开销
        torch.cuda.synchronize()
        print("GPU预热完成")
    
    # 创建模型
    model = create_model(config)
    
    return {
        'model': model,
        'dataset': dataset,
        'vocab': vocab,
        'itos': itos,
        'stoi': stoi,
        'config': config
    }