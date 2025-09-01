#!/bin/bash

echo "强制使用CPU模式运行Docker容器..."
echo "将src目录映射到容器的/app目录"

CURRENT_DIR=$(pwd)
echo "当前目录: $CURRENT_DIR"
echo "源代码目录: $CURRENT_DIR/src"

echo "启动CPU容器..."
docker run -it --rm \
    -v "$CURRENT_DIR/src:/app/src" \
    --name llm-0-1-cpu \
    llm-0-1:latest

if [ $? -ne 0 ]; then
    echo "容器运行失败！"
    echo "请确保："
    echo "1. Docker正在运行"
    echo "2. 镜像llm-0-1:latest已构建"
    echo "3. src目录存在"
    exit 1
fi

echo "容器已成功退出"
