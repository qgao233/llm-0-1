#!/bin/bash

echo "运行Docker容器..."
echo "将src目录映射到容器的/app目录"

CURRENT_DIR=$(pwd)
echo "当前目录: $CURRENT_DIR"
echo "源代码目录: $CURRENT_DIR/src"

# 检查是否支持GPU
GPU_SUPPORT=false
if docker run --rm --gpus all nvidia/cuda:11.7-base-ubuntu20.04 nvidia-smi >/dev/null 2>&1; then
    GPU_SUPPORT=true
    echo "检测到GPU支持，将启用GPU加速"
else
    echo "未检测到GPU支持或NVIDIA Container Toolkit未安装，使用CPU模式"
fi

# 根据GPU支持情况选择运行命令
if [ "$GPU_SUPPORT" = true ]; then
    echo "使用GPU模式启动容器..."
    docker run -it --rm \
        --gpus all \
        -v "$CURRENT_DIR/src:/app/src" \
        --name llm-0-1-container-gpu \
        llm-0-1:latest
else
    echo "使用CPU模式启动容器..."
    docker run -it --rm \
        -v "$CURRENT_DIR/src:/app/src" \
        --name llm-0-1-container-cpu \
        llm-0-1:latest
fi

if [ $? -ne 0 ]; then
    echo "容器运行失败！"
    echo "请确保："
    echo "1. Docker正在运行"
    echo "2. 镜像llm-0-1:latest已构建"
    echo "3. src目录存在"
    if [ "$GPU_SUPPORT" = true ]; then
        echo "4. NVIDIA Container Toolkit已正确安装"
        echo "5. GPU驱动程序已安装"
    fi
fi
