# Docker 使用说明

## 项目结构
```
llm-0-1/
├── src/                    # 源代码目录
│   └── index.py           # 主程序文件
├── requirements.txt       # Python依赖
├── Dockerfile.llm        # Docker构建文件
├── build.bat/.sh         # 构建脚本
├── run.bat/.sh           # 运行脚本
└── DOCKER_USAGE.md       # 使用说明
```

## 使用步骤

### 1. 构建Docker镜像

#### Windows系统：
```bash
build.bat
```

#### Linux/Mac系统：
```bash
chmod +x build.sh
./build.sh
```

### 2. 运行容器

#### Windows系统：
```bash
run.bat
```

#### Linux/Mac系统：
```bash
chmod +x run.sh
./run.sh
```

## 技术说明

### Dockerfile.llm 特点：
- 使用Python最新版本作为基础镜像
- 设置工作目录为 `/app`
- 安装requirements.txt中的依赖
- **源代码在运行时映射，不在构建时复制**

### 目录映射：
- 本地 `src/` 目录 → 容器 `/app` 目录
- 这样可以实时修改代码而无需重新构建镜像

### 容器运行参数：
- `-it`: 交互式运行
- `--rm`: 容器停止后自动删除
- `-v`: 卷映射，将本地src目录映射到容器/app目录
- `--name`: 容器名称为 llm-0-1-container

## 注意事项

1. 确保Docker服务正在运行
2. 确保src目录存在且包含index.py文件
3. 如果需要修改代码，直接编辑src目录下的文件即可，无需重新构建镜像
4. 如果修改了requirements.txt，需要重新构建镜像

## 故障排除

### 构建失败：
- 检查Docker是否正在运行
- 检查requirements.txt文件是否存在
- 检查网络连接（下载Python包需要网络）

### 运行失败：
- 检查镜像是否已构建：`docker images | grep llm-0-1`
- 检查src目录是否存在
- 检查Docker服务状态
