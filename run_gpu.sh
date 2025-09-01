#!/bin/bash

echo "强制使用GPU模式运行Docker容器..."
echo "将src目录映射到容器的/app目录"

CURRENT_DIR=$(pwd)
echo "当前目录: $CURRENT_DIR"
echo "源代码目录: $CURRENT_DIR/src"

# 检查NVIDIA Container Toolkit
echo "检查GPU支持..."
if ! docker run --rm --gpus all nvidia/cuda:11.7-base-ubuntu20.04 nvidia-smi >/dev/null 2>&1; then
    echo "错误：无法访问GPU或NVIDIA Container Toolkit未正确安装！"
    echo "请确保："
    echo "1. 安装了NVIDIA GPU驱动程序"
    echo "2. 安装了NVIDIA Container Toolkit"
    echo "3. Docker支持GPU加速"
    exit 1
fi

echo "GPU支持检查通过，启动GPU容器..."
docker run -it --rm \
    --gpus all \
    -v "$CURRENT_DIR/src:/app/src" \
    --name llm-0-1-gpu \
    llm-0-1:latest

if [ $? -ne 0 ]; then
    echo "容器运行失败！"
    echo "请确保："
    echo "1. Docker正在运行"
    echo "2. 镜像llm-0-1:latest已构建"
    echo "3. src目录存在"
    echo "4. GPU资源可用"
    exit 1
fi

echo "容器已成功退出"
