# Docker 使用说明

本项目提供了Docker容器化部署方案，支持CPU和GPU加速，方便在不同环境中运行。

## 项目结构
```
llm-0-1/
├── src/                    # 源代码目录
│   ├── llmcore.py         # 核心模型代码
│   ├── train.py           # 训练脚本
│   ├── infer.py           # 推理脚本
│   └── test_gpu.py        # GPU测试脚本
├── requirements.txt       # Python依赖
├── Dockerfile.llm        # Docker构建文件
├── build.bat/.sh         # 构建脚本
├── run.bat/.sh           # 自动检测运行脚本
├── run_gpu.bat/.sh       # 强制GPU运行脚本
├── run_cpu.bat/.sh       # 强制CPU运行脚本
└── DOCKER_USAGE.md       # 使用说明
```

## 构建镜像

使用提供的构建脚本：

**Windows:**
```bash
build.bat
```

**Linux/macOS:**
```bash
chmod +x build.sh
./build.sh
```

或者手动构建：
```bash
docker build -f Dockerfile.llm -t llm-0-1:latest .
```

## 运行容器

### 自动检测GPU/CPU模式

**Windows:**
```bash
run.bat
```

**Linux/macOS:**
```bash
chmod +x run.sh
./run.sh
```

这些脚本会自动检测GPU支持，如果检测到GPU和NVIDIA Container Toolkit，将自动启用GPU加速；否则使用CPU模式。

### 强制使用GPU模式

**Windows:**
```bash
run_gpu.bat
```

**Linux/macOS:**
```bash
chmod +x run_gpu.sh
./run_gpu.sh
```

### 强制使用CPU模式

**Windows:**
```bash
run_cpu.bat
```

**Linux/macOS:**
```bash
chmod +x run_cpu.sh
./run_cpu.sh
```

## GPU支持要求

要使用GPU加速，需要满足以下条件：

1. **NVIDIA GPU** - 支持CUDA 11.7或更高版本
2. **NVIDIA驱动程序** - 版本 >= 450.80.02
3. **Docker** - 版本 >= 19.03
4. **NVIDIA Container Toolkit** - 必须安装

### 安装NVIDIA Container Toolkit

**Ubuntu/Debian:**
```bash
# 添加GPG密钥
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# 安装nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 重启Docker服务
sudo systemctl restart docker
```

**Windows:**
使用Docker Desktop，确保在设置中启用GPU支持。

## 手动运行命令

### GPU模式
```bash
docker run -it --rm --gpus all -v "$(pwd)/src:/app/src" --name llm-0-1-gpu llm-0-1:latest
```

### CPU模式
```bash
docker run -it --rm -v "$(pwd)/src:/app/src" --name llm-0-1-cpu llm-0-1:latest
```

## 容器内使用

进入容器后，可以执行以下命令：

```bash
# 测试GPU支持
cd /app/src && python test_gpu.py

# 训练模型
cd /app/src && python train.py

# 推理
cd /app/src && python infer.py
```

## 目录映射

- 本地 `src/` 目录 → 容器 `/app/src` 目录
- 模型保存在 `src/model_save/` 目录中，会持久化到本地

## 故障排除

### GPU相关问题

1. **检查GPU是否可用:**
   ```bash
   nvidia-smi
   ```

2. **检查Docker GPU支持:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.7-base-ubuntu20.04 nvidia-smi
   ```

3. **检查容器内GPU:**
   ```bash
   docker run -it --rm --gpus all llm-0-1:latest python -c "import torch; print(torch.cuda.is_available())"
   ```

### 常见错误

- **"could not select device driver"** - NVIDIA Container Toolkit未安装
- **"CUDA out of memory"** - GPU内存不足，考虑减小batch_size
- **"No CUDA GPUs are available"** - GPU驱动或CUDA版本不兼容

### 构建失败：
- 检查Docker是否正在运行
- 检查requirements.txt文件是否存在
- 检查网络连接（下载Python包需要网络）

### 运行失败：
- 检查镜像是否已构建：`docker images | grep llm-0-1`
- 检查src目录是否存在
- 检查Docker服务状态

## 性能监控

在容器内监控GPU使用情况：
```bash
# 在另一个终端中运行
watch -n 1 nvidia-smi
```

## 技术说明

### Dockerfile.llm 特点：
- 使用PyTorch官方CUDA镜像作为基础镜像
- 预装CUDA 11.7和cuDNN 8
- 设置必要的NVIDIA环境变量
- 源代码在运行时映射，不在构建时复制

### 容器运行参数：
- `-it`: 交互式运行
- `--rm`: 容器停止后自动删除
- `--gpus all`: 启用所有GPU（GPU模式）
- `-v`: 卷映射，将本地src目录映射到容器/app/src目录

## 注意事项

1. 确保Docker已安装并运行
2. 首次运行前需要先构建镜像
3. 源代码通过卷映射到容器中，无需重新构建镜像即可更新代码
4. GPU模式下，模型训练速度会显著提升
5. 如果GPU内存不足，可以在代码中设置 `force_cpu=True` 强制使用CPU
6. 如果修改了requirements.txt，需要重新构建镜像