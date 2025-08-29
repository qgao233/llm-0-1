# 上面我们实现了LlaMa所需的算子，下面我们将其组合成为LlaMa的功能块，然后通过功能块的堆叠，组成LlaMa！

# 首先，设定一下功能块堆叠的层数：

# OK！ 现在我们更新一下，隐藏层维度堆叠多少层，我们先来4层尝尝咸淡！！！！
MASTER_CONFIG.update({
    'n_layers': 4,  
})

# 然后，组成LlaMa的功能块，实际上就是融合了RMS、AttentionRoPE、SwiGLU激活函数的那个简易模型：

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

# 看一下超参数字典

# 看一下我们的超参数字典
MASTER_CONFIG

# 输出：
#{'batch_size': 32,'context_window': 16, 'vocab_size': 4325,'d_model': 128,'epochs': 1000,'log_interval': 10,'n_heads': 8,'n_layers': 4}

# 测一下创建的功能块有没有问题：

# 用config字典，创建llama的功能块
block = LlamaBlock(MASTER_CONFIG)

# 生成一条随机数据，丢到这个llama功能块里，看一下是不是有bug
random_input = torch.randn(MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'], MASTER_CONFIG['d_model'])

# 执行以下看看输出
output = block(random_input)
output.shape

# 组装LlaMa！

# 现在，我们组装LlaMa
from collections import OrderedDict
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

# 训练LlaMa

# 开始训练咱们的Llama
llama = Llama(MASTER_CONFIG)
xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])
logits, loss = llama(xs, ys)
optimizer = torch.optim.Adam(llama.parameters())
train(llama, optimizer)

# 推理一下：

# 再看一下推理效果（实际上也没什么效果-。-）
# 别忘了generate里面的输入数据是咱们弄的5个0，如果替换为encode之后的数也是可以的！组成列表，转换tensor，这个应该没问题的吧~
generated_text = generate(llama, MASTER_CONFIG, 500)[0]
print(generated_text)


# 测试集跑一下，甚至都忘了测试集了吧，忘了的回头看看get_batch那个函数：

# 下面是测试集跑一下 
# 获取测试集的特征值和目标值
xs, ys = get_batches(dataset, 'test', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

# 丢进Llama获取loss
logits, loss = llama(xs, ys)

print(loss)
# 输出：
# tensor(4.7326, grad_fn=<NllLossBackward0>)

# 加点玩意，让其看上去更像回事，增加学习率调度器

# 还有优化的点哦，别忘了optimizer！以及学习率调度器！
# 调整参数再来一次！

MASTER_CONFIG.update({
    "epochs": 1000
})

# 学习率优化器选择余弦退火
llama_with_cosine = Llama(MASTER_CONFIG)

llama_optimizer = torch.optim.Adam(
    llama.parameters(),
    betas=(.9, .95),
    weight_decay=.1,
    eps=1e-9,
    lr=1e-3
)
# 余弦退火学习率优化器，让学习率逐渐减小，在结束时达到最低值。 详细可以百度，这种文章很多。
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(llama_optimizer, 300, eta_min=1e-5)

# 跑一下！
train(llama_with_cosine, llama_optimizer, scheduler=scheduler)


# 跑2万个epoch就好了。毕竟所有参数都是随便弄的。。。。。

# 好咯，剩下的没什么了，我们保存一下模型，然后再写个调用模型推理的函数。

# 参数路径自己改哦

# 保存模型：

# 保存模型权重
model_save_path = "./hf_model_save/pytorch_model.bin"
torch.save(llama_with_cosine.state_dict(), model_save_path)


# 生成一个config文件
import json

config_save_path = "./hf_model_save/config.json"
with open(config_save_path, 'w') as f:
    json.dump(MASTER_CONFIG, f)


# 保存optimizer和学习率调度器的状态，方便继续微调
optimizer_save_path = "./hf_model_save/optimizer.pt"
torch.save(llama_optimizer.state_dict(), optimizer_save_path)

scheduler_save_path = "./hf_model_save/scheduler.pt"
torch.save(scheduler.state_dict(), scheduler_save_path)

# 加载模型：

# 接下来是加载模型
llama_with_cosine = Llama(MASTER_CONFIG)  

# 加载模型权重
model_save_path = "./hf_model_save/pytorch_model.bin"
llama_with_cosine.load_state_dict(torch.load(model_save_path))

# 设置为评估模式
llama_with_cosine.eval()


# 加载优化器和学习率调度器，如果需要继续训练什么的。
llama_optimizer.load_state_dict(torch.load(optimizer_save_path))
scheduler.load_state_dict(torch.load(scheduler_save_path))


# 进行推理
output = generate(llama_with_cosine, MASTER_CONFIG)
print(output)


# meta llama源码：https://link.zhihu.com/?target=https%3A//github.com/meta-llama/llama3/blob/main/llama/model.py


