# 将LlaMa的算子加入到上面的简易模型架构`seq2seq_base.py`中

主要包括：

- RMS_Norm
- RoPE
- SwiGLU

## RMSNorm快速了解

norm，做标准化，训练过程中的张量标准化操作，通过计算均值和方差，将样本进行归一化。 

在大学课程《概率与统计》我们学过，样本的均值代表样本的特征，而方差代表离散程度。

因此，通过计算，让数据变为均值为0，方差为1的数据。 这样可以使数据服从标准的正态分布。

记得大学时候，老师讲这一段的时候，着重强调：“高斯分布，正态分布”，也可以叫自然分布，自然界的很多统计情况，几乎都满足高斯分布。 两边向中心靠拢，超过中心的，随着逐渐增大，会越来越少，没超过中心的，距离中心越远，数量也越来越少。而分布的众数永远都是在中间。

啊~ 数学之美，但是也美不过高数老师（嘿嘿）。

使用均值和方差计算数据的标准差，这样既保留了数据的异常值，同时维持数据的异常结构，这样可以稳定梯度，让梯度变化更稳定，减少梯度消失或者爆炸的问题，因为维持了异常结构，也能减少过拟合问题，增强泛化能力。



RMSNorm出来之前，广泛使用的batch_normlize，针对批次数据做标准化。标准化的数值是一个batch作为一个样本总体，计算其均值与方差。



而后，又出现了layer_norm，其是针对每个token的特征向量做归一化处理（不知道特征向量，请看本人之前的rope文章。应该可以理解token和特征向量的关系。）依旧需要计算均值和方差。



RMSNorm和layer_norm的主要区别在于RMSNorm不需要同时计算均值和方差两个统计量，而只需要计算均方根这一个统计量。在模型表现效果几乎与layer_norm持平的前提下，节省7%-64%的计算量。

RMS_Norm计算公式：

\[ \text{RMS}(x) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2} \]


**RMSNorm 的工作原理**
- 计算 RMS： 对于输入的特征（如神经元的输出），首先计算其 RMS 值。
- 标准化： 将每个特征的值除以其 RMS 值，从而调整数据的尺度，使其均值为 0，标准差为 1。
- 缩放和平移： 在归一化后，RMSNorm 通常还会引入可学习的参数（缩放因子和偏置），以便模型能够学习适合特定任务的特征表示。

猜想： 既然都平方根了，突然想起那个让程序员之神--约翰·卡马克直呼“卧槽”的快速平方根倒数算法了。 当然，也有可能这么经典的数值计算方法已经被集成进了pytorch。

RMS基本介绍差不多了，下面开始实现RMSNorm模块：

```python
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

        # 返回缩放后归一化的张量
        # print(self.scale[:x.shape[1], :].unsqueeze(0) * raw)
        return self.scale[:x.shape[1], :].unsqueeze(0) * raw
```

需要注意的是，`raw = x / ff_rms.unsqueeze(-1).unsqueeze(-1)`这里的计算是针对张量矩阵中的每个元素计算，计算出了范数（实际上就是一个数值，不是矩阵），对原本输入数据的张量矩阵中，每一个元素除以这个归一化范数。因为RMSNorm已经继承在pytorch官方的nn.Moudle中，因此我们对其稍加修改。

接着：我们将这个RMS_Norm算子加入到上面的简易模型中，代码中很清楚，如何加入的：

```python
class SimpleNotStupidModel_RMS(nn.Module):
    def __init__(self, config=MASTER_CONFIG):
      super().__init__()
      self.config = config
      self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
      # 在这里，我们添加RMS层
      self.rms = RMSNorm((config['context_window'], config['d_model']))
      self.linear = nn.Sequential(
          nn.Linear(config['d_model'], config['d_model']),
          nn.ReLU(),
          nn.Linear(config['d_model'], config['vocab_size']),
      )
      print("Model parameters:", sum([m.numel() for m in self.parameters()]))

    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        # 在这里，添加实例化后的RMS层，承接Embedding层输出的张量
        x = self.rms(x)

        logits = self.linear(x)
        # print(logits.shape)

        if targets is not None:

            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss
        else:
            return logits
        print("Model parameters:", sum([m.numel() for m in self.parameters()]))
```

我们添加了RMS_Norm之后，再看一下训练效果：

```python
# 好啦，这样我们对原来的NotStupidModel添加了RMSNorm，现在执行一下看看
model = SimpleNotStupidModel_RMS(MASTER_CONFIG)

xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

logits, loss = model(xs, ys)

optimizer = torch.optim.Adam(model.parameters())

train(model, optimizer)

# 在同样的训练超参数设置上，加入了RMSNorm的训练速度明显加快。
```

## 将旋转位置编码（Rope）加入到模型中

RoPE很巧妙的将笛卡尔坐标系的计算，拉到极坐标空间计算。具体原理快速过一下可能说不完，参考：

https://zhuanlan.zhihu.com/p/780744022
https://zhuanlan.zhihu.com/p/830878252


先定义一个函数，用于计算旋转位置编码：

```python
def get_rotary_matrix(context_window, embedding_dim):
    # 初始化一个0填充，形状为（context_window, embedding_dim, embedding_dim）的张量矩阵，其中context_window为token数量，后面两个embedding_dim组成正方形矩阵，与后面的attention计算对齐格式
    R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
    
    # 遍历每一个位置的token
    for position in range(context_window):
        # 还记得我的上一篇文章中说的，对于特征，两两组合吗，因此需要循环的次数为embedding_dim除以2
        for i in range(embedding_dim // 2):
            # 设置θ值，采样频率，或者说旋转频率，旋转角都可以，除以embedding_dim防止梯度问题。(此处由知乎ID为：bignova的大佬修正。
            theta = 10000. ** (-2. * i) / embedding_dim)
            # 根据欧拉公式，计算旋转的角度，分别有sin 和cos，将计算拉到复数空间，并将旋转角度应用在上面的0填充的矩阵
            m_theta = position * theta
            R[position, 2 * i, 2 * i] = np.cos(m_theta)
            R[position, 2 * i, 2 * i + 1] = -np.sin(m_theta)
            R[position, 2 * i + 1, 2 * i] = np.sin(m_theta)
            R[position, 2 * i + 1, 2 * i + 1] = np.cos(m_theta)
            # 得到的结果是旋转位置编码矩阵，到这里还没覆盖到attention
    return R
```

因为旋转位置编码是结合了注意力机制的Q和K进行计算的，因此我们先实现一个单头的注意力机制的算子：

```python
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
        # 获得旋转位置编码矩阵，接下来会覆盖Q和K权重矩阵
        self.R = get_rotary_matrix(config['context_window'], config['d_model'])


    # 这里将上一个代码块中实现的创建旋转位置编码的功能函数原封不动的拿过来
    def get_rotary_matrix(context_window, embedding_dim):
        # 初始化一个0填充，形状为（context_window, embedding_dim, embedding_dim）的张量矩阵，其中context_window为token数量，后面两个embedding_dim组成正方形矩阵，与后面的attention计算对齐格式
        R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
        
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

    def forward(self, x, return_attn_weights=False):
        # 前向传播时，输入矩阵的形状为(batch, sequence length, dimension)

        b, m, d = x.shape  # batch size, sequence length, dimension

        # 线性变换Q,K,V
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # 将旋转位置编码应用于Q和K，其中torch.bmm为矩阵做内积，transpose是转置，对Q矩阵转置，并与旋转位置编码做内积，再转置回原状，Q便应用了旋转位置编码。
        # 考虑到输入文本的长度，因此对位置编码矩阵在第一维度做截断，因为长了也没用，与文本长度一样。
        q_rotated = (torch.bmm(q.transpose(0, 1), self.R[:m])).transpose(0, 1)
        # 同理对K也应用旋转位置编码进行覆盖
        k_rotated = (torch.bmm(k.transpose(0, 1), self.R[:m])).transpose(0, 1)

        # 对注意力机制点积进行等比例缩放，防止attention张量过长引发梯度爆炸，对应
        activations = F.scaled_dot_product_attention(
            q_rotated, k_rotated, v, dropout_p=0.1, is_causal=True
        )
        # 如果return_attn_weights参数置为1，则需要对attention进行掩码，因为在学习的时候，希望模型能依据前n个token去预测token，而不是开卷考试。
        if return_attn_weights:
            # 创建注意力掩码矩阵，其中torch.tril函数为：对于矩阵，取左下三角，剩下的都置0
            attn_mask = torch.tril(torch.ones((m, m)), diagonal=0)
            # 计算注意力机制的权重矩阵，并对最后一维度做归一化，（突击检查）为什么是最后一维！因为最后一维度是每个token的特征向量！
            attn_weights = torch.bmm(q_rotated, k_rotated.transpose(1, 2)) / np.sqrt(d) + attn_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            return activations, attn_weights

        return activations
```


单头注意力机制实现完毕，下面是多头的注意力机制：

```python
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
        heads = [h(x) for h in self.heads]
        # 输入张量x经过多个头计算attention（同时，attention是已经覆盖了RoPE的），重新拼接成新的矩阵，重新放入变量x。到这里你应该觉得：那矩阵形状不就变了吗
        x = torch.cat(heads, dim=-1)
        
        # 这不，线性层的作用来了
        x = self.linear(x)
        
        # 随机失活一下，防止过拟合
        x = self.dropout(x)
        return x
```

如果已经感觉，除了懵逼，已经伤脑了。还是先学习一下RoPE的原理，即使不是本厮的那两篇，也可以找一些能理解的文章，白话解释，甚至问chatgpt，任何方法，只要作者的讲述方式，和自己的脑电波频率对得上，能理解，就OK。

好啦，上面的功能已实现，让我们更新一下config超参数的字典，设定注意力机制头数，LlaMa是32个注意力机制头，我们创建8个：

```python
MASTER_CONFIG.update({
    'n_heads': 8,
})
```

接下来，我们更新我们的建议模型，之前更新了RMS_Norm，这次我们在那基础上，将融合了ROPE位置编码的多头注意力机制加入进去！

```python
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
```

老规矩，跑一下，看看效果！

```python
# 再跑一下！
model = RopeModel(MASTER_CONFIG)
xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])
logits, loss = model(xs, ys)
optimizer = torch.optim.Adam(model.parameters())
train(model, optimizer)
```

> loss没怎么下降，因为attention没有做mask，虽然功能实现了，但是没有调用，布尔值设置的是false。

### SwiGLU算子

将swish和glu结合起来。这两个激活函数单独拿出来都很强，结合起来。

这玩意挺玄学，说它不好吧，但是这玩意确实比relu这败家子保留更多语义特征参数，不至于某个权重突然到了小于0的区间，然后糊里糊涂的消失。说它好吧，它的计算量确实挺大。

swish用了sigmoid，GLU用了门控结构（门控结构思想，可以学习一下RNN,GRU,LSTM什么的）

**由于SwiGLU部分，本厮也没深入研究过，因此以下是抄来的东西：**

```
Swiglu 的工作原理
Swiglu 将 Swish 和 GLU 结合在一起，计算方式如下：
计算两个部分：
输入通过线性变换得到 a 和 b。
a 使用 Swish 激活，b 使用 Sigmoid 激活。
组合结果：
最终的输出由 a 和 b 的组合决定，形式为： 
优势
灵活性：Swiglu 结合了非线性和门控机制，允许模型更灵活地捕捉复杂的模式。
性能：在某些任务上，Swiglu 已显示出比传统激活函数更好的表现，尤其是在处理复杂数据时。
```

实现一下：

```python
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
```

再将刚刚定义好的swiglu算子添加到模型中，该模型我们之前添加过RMS、RopeAttention。

```python
# 再将swiglu添加进上面的模型
class RopeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.rms = RMSNorm((config['context_window'], config['d_model']))
        self.rope_attention = RoPEMaskedMultiheadAttention(config)
        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            # 在这里，增加了SwiGLU层
            SwiGLU(config['d_model']),  
        )
        self.last_linear = nn.Linear(config['d_model'], config['vocab_size'])
        print("model params:", sum([m.numel() for m in self.parameters()]))

    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        x = self.rms(x)  
        x = x + self.rope_attention(x)
        x = self.rms(x)
        x = x + self.linear(x)
        logits = self.last_linear(x)

        if targets is not None:
            # Calculate cross-entropy loss if targets are provided
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss

        else:
            return logits
```

再跑一次！

```python
# 一二三四！再来一次！
model = RopeModel(MASTER_CONFIG)
xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])
logits, loss = model(xs, ys)
optimizer = torch.optim.Adam(model.parameters())
train(model, optimizer)
```