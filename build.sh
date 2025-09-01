#!/bin/bash

echo "开始构建Docker镜像..."
echo "使用Dockerfile.llm构建镜像"

docker build -f Dockerfile.llm -t llm-0-1:latest .

if [ $? -eq 0 ]; then
    echo "镜像构建成功！"
    echo "镜像名称: llm-0-1:latest"
    echo "可以使用 ./run.sh 脚本运行容器"
else
    echo "镜像构建失败！"
    echo "请检查Docker是否正在运行"
fi
