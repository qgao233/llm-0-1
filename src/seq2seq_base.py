# 参考：https://zhuanlan.zhihu.com/p/1674261485


# 构造一个简单的文本生成模型
# 在构造LlaMa之前，我们先构造一个简单的seq2seq模型，然后逐步对原本的Seq2seq模型，增加LlaMa中的算子RMS、Rope、SwiGLU，直到完整构造LlaMa。

# 首先是一些功能函数的实现，虽然没什么难的，但是最好还是 过一遍，因为脑海里有数据的形状，在模型搭建的时候，知道输入进去的是什么样子的，对于理解深度神经网络有很大帮助。

# 导包

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd
import urllib.request

# 创建一个字典用于存储config

MASTER_CONFIG = {
    # 参数放这里
}

url = "https://raw.githubusercontent.com/mc112611/PI-ka-pi/main/xiyouji.txt"
file_name = "xiyouji.txt"
urllib.request.urlretrieve(url, file_name)

lines = open("xiyouji.txt", 'r').read()

# 创建简易版词表（字符级）
vocab = sorted(list(set(lines)))

# 查看词表前n个字符
head_num=50
print('词表前{}个:'.format(head_num), vocab[:head_num])

print('词表大小:', len(vocab))

# 输出：
# 词表前50个: ['\n', ' ', '!', '"', '#', '*', ',', '.', '—', '‘', '’', '“', '”', '□', '、', '。', '《', '》', '一', '丁', '七', '万', '丈', '三', '上', '下', '不', '与', '丑', '专', '且', '丕', '世', '丘', '丙', '业', '丛', '东', '丝', '丞', '丢', '两', '严', '丧', '个', '丫', '中', '丰', '串', '临']
# 词表大小: 4325

# ------------------------------------------------------------

# 将词表编码成为数字，普通的整数，如："1":"孙"，"2":"悟" ，"3":"空"，下面那个是键和值对调i

itos = {i: ch for i, ch in enumerate(vocab)}

# 双向映射
stoi = {ch: i for i, ch in enumerate(vocab)}

# ------------------------------------------------------------

# 接下来我们创建一个简易的编码器和解码器。用于输入进模型之前，将文本转换为数字，输出之前将数字转换为文本：
# 编码器（青春版）
def encode(s):
    return [stoi[ch] for ch in s]

# 解码器（青春版）
def decode(l):
    return ''.join([itos[i] for i in l])

# 来试一下这个“高端”的编解码器
decode(encode("悟空"))
encode("悟空")

# 输出：
# [1318, 2691]
# 输出的两个数字代表着“悟”字和“空”字，在vocab词表中的编码，映射表的原理

# 因为是字符级的编码，不是BPE，或者其他格式的词表构建。  此部分，可以替换为tiktoken或者其他的映射器，本文主要以理解原理为主，使用的方式越简单越容易理解。

# ------------------------------------------------------------

# 将整本《西游记》全部编码，转换成tensor。本文一共有65万多个字符，转换成一维的张量。

# 对全文进行编码，并映射成为tensor
dataset = torch.tensor(encode(lines), dtype=torch.int16)

# 看一下形状，实际上就是多少个字符，一共65万个字符
print(dataset.shape)
print(dataset)

# 输出：
# torch.Size([658298])
# tensor([   0, 4319, 1694,  ...,   12,    0,    0], dtype=torch.int16)


# ------------------------------------------------------------

# 接下来构建batch

def get_batches(data, split, batch_size, context_window, config=MASTER_CONFIG):
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

    # 这里需要学习torch.randint，生成大小为batch_size，内部数值为随机整数的tensor。生成随机数的数值域为[0,训练集字符数量-滑动窗口大小-1]之间的整数
    # 详情可以参考官方文档，或者这个博客：https://blog.csdn.net/qq_41813454/article/details/136326473
    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,))
    # print('ix输出:')
    # print(ix)


    # 这里需要学习torch.stack，执行操作类似于python的zip关键字，只不过操作对象是tensor张量，指定任意维度的张量进行组合
    # 详情参考官方文档，或者这个博客：https://blog.csdn.net/dongjinkun/article/details/132590205

    # 这里x作为特征，y作为预测值，因为文本生成任务是根据前n个字符，去推理后面的1个字符，因此y的构造会使窗口在保持原大小的基础上向后移一位
    # 通过滑动窗口，对batch_data中的训练数据，进行随机取样，相当于随机选择训练数据。
    # 在原65万多个字符中，随机选取一个字符作为开始，并以这个开始点，向后选取滑动窗口个数的字符，作为训练数据，向后移一位就是其目标值。  因此ix的构造不能超出index。
    x = torch.stack([batch_data[i:i+context_window] for i in ix]).long()
    y = torch.stack([batch_data[i+1:i+context_window+1] for i in ix]).long()

    # 返回特征值，目标值
    return x, y

# torch.randint以及 torch.stack  需要学习一下，这种的文章有很多。
# 重点可以放在这两行代码上：    
# x = torch.stack([batch_data[i:i+context_window] for i in ix]).long() 
# y = torch.stack([batch_data[i+1:i+context_window+1] for i in ix]).long()  
# 以滑动窗口的形式对数据进行获取特征值与目标值，因为训练的时候，是根据前N个字符去预测后面的1个字符，因此，目标值y的滑动窗口在保持原本大小（长度）的前提下，向右移动一位。


# ------------------------------------------------------------

# 根据上面构造的get_batchs()函数，更新参数字典。
MASTER_CONFIG.update({
    'batch_size': 8,          # 不解释
    'context_window': 16,      # 滑动窗口采样，设置采样大小
    'vocab_size':4325         # 咱们的西游记数据集，一共包含4325个不重复的汉字，标点符号
})

# 构造一个字典，用于存储config参数，滑动窗口的值16，代表着采样时候将每条文本，分成多个长度为16的数据。vocab_size，上面有，代表着《西游记》原文所有字符去重后字符数量，也就是词表的大小。

# ------------------------------------------------------------

# 为了方便理解，我们每构造一个函数，或者类，就单独运行一下，看看效果。 如果堆了很多代码，即使执行，看到最终结果，应该也是懵的。

# 所以我们执行刚刚的get_ batches函数，看一下结果，为了方便观察，使用解码器（青春版）解一下码。

# 获取训练数据
xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

# 因为是随机生成的采样，我们可以看一下数据，其中每个采样数据，来自于原文随机的起始点，每个元组为一个（x,y），可以观察每个x和y的首位去直观感受一下滑动窗口执行的操作
decoded_samples = [(decode(xs[i].tolist()), decode(ys[i].tolist())) for i in range(len(xs))]

print(decoded_samples)

# 输出：
# [('姿娇且嫩’ ！”那女子笑\n而悄答', '娇且嫩’ ！”那女子笑\n而悄答道'), 
# ('泼猢狲，打杀我也！”沙僧、八戒问', '猢狲，打杀我也！”沙僧、八戒问道'), 
# ('人家，不惯骑马。”唐僧叫八戒驮着', '家，不惯骑马。”唐僧叫八戒驮着，'), 
# ('著一幅“圯桥进履”的\n画儿。行者', '一幅“圯桥进履”的\n画儿。行者道'), 
# ('从何来？这匹马，他在此久住，必知', '何来？这匹马，他在此久住，必知水'), 
# ('声去，唿哨一声，寂然不见。那一国', '去，唿哨一声，寂然不见。那一国君'), 
# ('刀轮剑砍怎伤怀！\n火烧雷打只如此', '轮剑砍怎伤怀！\n火烧雷打只如此，'), 
# ('鲜。紫竹几竿鹦鹉歇，青松数簇鹧鸪', '。紫竹几竿鹦鹉歇，青松数簇鹧鸪\n')]

# 输出结果结果是列表，列表中的元素为多个元组，每个元组是特征数据与目标值。可以观察每个元组数据的开始和结束，来看滑动窗口的效果。

# ------------------------------------------------------------

# 接下来构造一个评估函数

# 推理时禁用梯度计算
@torch.no_grad()
def evaluate_loss(model, config=MASTER_CONFIG):
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
            xb, yb = get_batches(dataset, split, config['batch_size'], config['context_window'])

            # 把拿到的数据丢进模型，得到loss值
            _, loss = model(xb, yb)

            # 更新loss存储
            losses.append(loss.item())

        # 这里就是大家经常在控制台看到的 "train_loss"  "valid_loss"由来
        out[split] = np.mean(losses)

    # 评估完了，别忘了把模型再置回训练状态，下一个epoch还要继续训练呢
    model.train()

    return out

# 到这里应该没有令人“眼前一黑”的操作。

# 这里加一个分割线，没什么特别的作用，就是觉得，如果上面的内容过了一遍，在jupyter或者colab跑了一下，没啥问题，那么可以继续。

# 说一些题外的，如果想抱着，代码拿来就能用，这篇文章可能没那么神（毕竟一个西游记原文，65万字的数据，还没有预训练模型，能练出个什么惊天大模型出来）。如果是想学习LLM，那么再提醒一次，请认真逐行看一下代码，否则会懵逼且伤脑。

# ------------------------------------------------------------


# 在进行分析LlaMa架构分析之前，我们从最简单的文本生成模型开始创建，然后在最简单的文本生成模型的基础上，把LlaMa的RSM，Rope等一点点添加进去。为此我们先：

# 创建一个有毛病的模型架构
# 分析一下这个架构（其实也没什么分析的）

class StupidModel(nn.Module):
    def __init__(self, config=MASTER_CONFIG):
        super().__init__()
        self.config = config

        # embedding层，输入：词表大小，输出：维度大小
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])

        # 创建线性层用于捕捉特征关系
        # 下面突击检查：这玩意是不是隐藏层！线性层堆叠越多是不是越好！堆叠越多是不是更计算开销越大！
        # LlaMa使用的激活函数是SwiGLU，目前在这个Stupid模型架构里面先用Relu
        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.ReLU(),
            nn.Linear(config['d_model'], config['vocab_size']),
        )

        # 这个命令可以背一下，或者复制粘贴到自己的学习笔记。 因为这行命令会直接帮你查看模型的参数量。
        # 否则要么自己手算，要么就是听别人讲某某模型 7B  20B  108B   有了这个命令，你就能直接查看你创建的模型参数量多少
        print("模型参数量：", sum([m.numel() for m in self.parameters()]))


# 我想，看注释就可以了吧,我们创建了一个有些stupid的模型，结构很简单：嵌入，线性变换，激活函数。唯一比较有意思的是，记住下面这个命令，以后可以用来查参数量。

# print("模型参数量：", sum([m.numel() for m in self.parameters()]))

# ------------------------------------------------------------

# 接下来，我们对上面那个斯丢匹得模型增加前向传播函数，也就是forward，换个名字，叫：“简单的破模型”吧。 实际上broken代表着有问题。至于问题，马上解答。

class SimpleBrokenModel(nn.Module):
    # init里的跟上面一样，没变化
    def __init__(self, config=MASTER_CONFIG):
      super().__init__()
      self.config = config
      self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
      self.linear = nn.Sequential(
          nn.Linear(config['d_model'], config['d_model']),
          nn.ReLU(),
          nn.Linear(config['d_model'], config['vocab_size']),
      )



      # 添加前向传播函数
    def forward(self, idx, targets=None):
        # 实例化embedding层，输入映射为id的数据，输出嵌入后的数据
        x = self.embedding(idx)

        # 线性层承接embedding层输出的数据
        a = self.linear(x)

        # 对线性层输出的数据在最后一个维度，做softmax，得到概率分布
        logits = F.softmax(a, dim=-1)

        # 如果有目标值（也就是我们前面的y），则计算通过交叉熵损失计算loss结果。给输出的概率矩阵变个形状，再给目标值变个形状。  统一一下输入输出，然后计算loss。其中最后一维代表着一条数据。
        # 此处需要了解tensor.view()函数，带上几何空间想象力去想一下矩阵的形状。
        if targets is not None:

            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss

        # 如果没有目标值，则只返回概率分布的结果
        else:
            return logits

        # 查看参数量
        print("模型参数量：", sum([m.numel() for m in self.parameters()]))

# 注释里不只有训练阶段，同时也有推理（评估）阶段，条件判断目标值是否存在，如果不存在，则只输出模型输出的结果，如果目标值存在，则还要返还loss值。

# 这里我们设置这个模型为128维的embedding
MASTER_CONFIG.update({
    'd_model': 128,
})

# 实例化模型，传参
model = SimpleBrokenModel(MASTER_CONFIG)

# 再看看参数量
print("咱们的模型这么多参数量:", sum([m.numel() for m in model.parameters()]))
# 于是乎，我们创建了一个1128307个参数的模型，上面参数想怎么改，自己改！电脑不会爆炸！

# LlaMa的嵌入维度是4096，我们弄个小一点的，毕竟在CPU上训练，我们设置嵌入维度为128，得到一个112万参数量的模型。

# ------------------------------------------------------------

# 下面，查看一下没训练之前，模型输出的结果，以及loss，模型输出的结果也就是数字，看loss更直观一些

# 获取训练的特征数据与目标数据
xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

# 扔进模型获取概率分布矩阵与loss
logits, loss = model(xs, ys)
loss

# 输出：
# tensor(8.3722, grad_fn=<NllLossBackward0>)


# ------------------------------------------------------------

# 接下来，更新一下config超参数，实例化模型，以及设置优化器：

# 更新参数，训练伦次，batch_size，log日志打印步长
MASTER_CONFIG.update({
    'epochs': 1000,
    'log_interval': 10,      # 每10个batch打印一次log
    'batch_size': 32,
})

# 实例化模型
model = SimpleBrokenModel(MASTER_CONFIG)

# 创建一个Adam优化器，基础知识，
optimizer = torch.optim.Adam(
    model.parameters(),      # 优化器执行优化全部的模型参数
)

# 再构造一个训练函数，并启动训练：

# 构建训练函数
def train(model, optimizer, scheduler=None, config=MASTER_CONFIG, print_logs=False):
    # loss存储
    losses = []

    # 训练时间记录开始时间
    start_time = time.time()

    # 循环训练指定epoch的轮数
    for epoch in range(config['epochs']):
        # 优化器要初始化啊，否则每次训练都是基于上一次训练结果进行优化，效果甚微
        optimizer.zero_grad()

        # 获取训练数据
        xs, ys = get_batches(dataset, 'train', config['batch_size'], config['context_window'])

        # 前向传播计算概率矩阵与loss
        logits, loss = model(xs, targets=ys)

        # 反向传播更新权重参数，更新学习率优化器
        loss.backward()
        optimizer.step()

        # 如果提供学习率调度器，那么学习率会通过调度器进行修改，比如学习率周期性变化，或者梯度减小，增加，具体策略需要综合考虑进行设置，详情自行查询，关键字：lr_scheduler
        if scheduler:
            scheduler.step()

        # 打印log
        if epoch % config['log_interval'] == 0:
            # 训练时间
            batch_time = time.time() - start_time

            # 执行评估函数，在训练集和验证集上计算loss
            x = evaluate_loss(model)

            # Store the validation loss
            losses += [x]

            # 打印进度日志
            if print_logs:
                print(f"Epoch {epoch} | val loss {x['val']:.3f} | Time {batch_time:.3f} | ETA in seconds {batch_time * (config['epochs'] - epoch)/config['log_interval'] :.3f}")

            # 重置开始时间，用于计算下一轮的训练时间
            start_time = time.time()

            # 打印下一轮的学习率，如果使用了lr_scheduler
            if scheduler:
                print("lr: ", scheduler.get_lr())

    # 上面所有epoch训练结束，打印最终的结果
    print("Validation loss: ", losses[-1]['val'])

    # 返还每一步loss值的列表，因为我们要画图，返还的是loss迭代的图像
    return pd.DataFrame(losses).plot()

# 启动训练
train(model, optimizer)



# 输出的loss变化是这样的：
# ![[Pasted image 20250829145240.png]]

# 在1000个epoch的训练后，loss值从原来的8.3722，下降到了8.2150！

# 所以知道为什么这个模型架构叫“stupid”了吧，或者说“Broken“。

# 先分析一下原因：

# 上面那个训练框架存在一些问题。 回到前向传播的代码，也就是forward()中。 我们使用了 logits = F.softmax(a, dim=-1) 对线性层输出的结果做了一次概率分布的计算。 而loss的计算选择了交叉熵损失， 目标值的词表映射结果是整数，而模型输出的logits是概率矩阵。

# 这俩玩意计算，相当于在国内的大街上，遇到外国人，跟人家说一句“hey man,what's up”，老外说：“我会中文，嘎哈呀“，这么整也没问题，但是收敛起来就比较麻烦。

# 为了使loss计算更精确，我们需要将softmax去除。 以保证交叉熵损失的计算效果更好。

# 拿掉softmax，logits改为获取最后一个线性层输出的结果，不进行softmax计算概率分布。
# 因此将这个架构取名为：不那么蠢的模型架构
class SimpleNotStupidModel(nn.Module):
    def __init__(self, config=MASTER_CONFIG):
      super().__init__()
      self.config = config
      self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
      self.linear = nn.Sequential(
          nn.Linear(config['d_model'], config['d_model']),
          nn.ReLU(),
          nn.Linear(config['d_model'], config['vocab_size']),
      )
      print("Model parameters:", sum([m.numel() for m in self.parameters()]))

    def forward(self, idx, targets=None):
        x = self.embedding(idx)

        # 看这里，线性层直接输出结果，不转换为概率矩阵，只修改这里，其余不动。
        logits = self.linear(x)
        # print(logits.shape)

        if targets is not None:

            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss
        else:
            return logits
        print("Model parameters:", sum([m.numel() for m in self.parameters()]))

# 再跑一下：

# 再来一次实例化各种功能，再启动一次训练
model = SimpleNotStupidModel(MASTER_CONFIG)
xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])
logits, loss = model(xs, ys)
optimizer = torch.optim.Adam(model.parameters())
train(model, optimizer)

# loss开窍了，下降了很多到5点几

# ------------------------------------------------------------

# 接下来构造一个推理函数：

# 并不准备传入数据，学习版，咱们就弄5个0，也就是词表中的换行符'\n'去推理。以换行符开始，循环推测后面20个字符

# 推理函数（输出结果就别纠结其效果了，权重都没保存，就是根据模型初始化生成的随机数组成的矩阵做的推理）
def generate(model, config=MASTER_CONFIG, max_new_tokens=20):
    # 生成5个0，作为输入数据,5行一列，代表输入5个字符。 这个地方可以自行替换其他随机数测试。
    idx = torch.zeros(5, 1).long()
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
    return [decode(x) for x in idx.tolist()]

generate(model)

# 输出结果，不出所料，合乎情理，意料之中，挺差的:

# ['\n尽行者，咬道：“尽人了卖。”这人。”众僧',
#  '\n揪啊？怎么整猜脸。”那怪教沙僧护菩萨，貌',
#  '\n你看着国尖，不知请你。”妖王髯明。\n须行',
#  '\n老朗哥啊，遇货出便？路，径出聋送做者，似',
#  '\n那个急急的八戒、沙僧的树。菩萨莲削，身\n']

# OK，到这里构造一个简单的Seq2seq模型，完毕，还是希望对上面的代码跑一下，理解一下，按行自行分析一下，如果有搞不清楚的，打印一下shape，看一下数据形状如何变换的。

# 如果上面全部搞清楚了，下面开始进入正题：在上面简单的模型基础上，逐步增加LlaMa的算子，并且每增加一个算子就看一下效果。

# 代码依旧是逐行注释，并且有些地方，咱也是为了学习分享，也记录了一些想法。