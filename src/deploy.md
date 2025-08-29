# 接下来可以整点花活儿，比如：部署一个异步的远程服务

```sh
!pip install fastapi uvicorn
```

> 请注意，! 符号在 Unix-like 系统的 shell 中用于否定前一个命令的结果，但在这里它不是必需的，可以省略。

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F

# 初始化 FastAPI
app = FastAPI()

# 模型加载
model_path = "./hf_model_save/pytorch_model.bin"
model = Llama(MASTER_CONFIG)
model.load_state_dict(torch.load(model_path))
model.eval()

class InputData(BaseModel):
    idx: list

@app.post("/generate/")
async def generate(model, config=MASTER_CONFIG, max_new_tokens=20):
    # 生成随机数，作为输入数据,5行一列，代表输入5个字符。 这个地方可以自行替换其他随机数测试。
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
```

```python
# 在colab里启动还是挺麻烦的。  建议把所有代码整理一下，在服务器，或者个人电脑里运行
import nest_asyncio
import uvicorn

nest_asyncio.apply()

# 启动 FastAPI 应用
uvicorn.run(app, host="0.0.0.0", port=8000)
```

```python
# 服务部署成功后，可以发送请求测试效果
import requests

input_data = {"idx": [[0]]}  # 根据需求提供输入数据
response = requests.post("http://localhost:8000/generate/", json=input_data)
print(response.json())
```