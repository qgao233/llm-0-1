@echo off
setlocal enabledelayedexpansion

echo 运行Docker容器...
echo 将src目录映射到容器的/app目录

set CURRENT_DIR=%cd%
echo 当前目录: %CURRENT_DIR%
echo 源代码目录: %CURRENT_DIR%\src

:: 检查是否支持GPU
set GPU_SUPPORT=false
docker run --rm --gpus all nvidia/cuda:11.7-base-ubuntu20.04 nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set GPU_SUPPORT=true
    echo 检测到GPU支持，将启用GPU加速
) else (
    echo 未检测到GPU支持或NVIDIA Container Toolkit未安装，使用CPU模式
)

:: 根据GPU支持情况选择运行命令
if "!GPU_SUPPORT!"=="true" (
    echo 使用GPU模式启动容器...
    docker run -it --rm ^
        --gpus all ^
        -v "%CURRENT_DIR%\src:/app/src" ^
        --name llm-0-1-container-gpu ^
        llm-0-1:latest
) else (
    echo 使用CPU模式启动容器...
    docker run -it --rm ^
        -v "%CURRENT_DIR%\src:/app/src" ^
        --name llm-0-1-container-cpu ^
        llm-0-1:latest
)

if %ERRORLEVEL% NEQ 0 (
    echo 容器运行失败！
    echo 请确保：
    echo 1. Docker正在运行
    echo 2. 镜像llm-0-1:latest已构建
    echo 3. src目录存在
    if "!GPU_SUPPORT!"=="true" (
        echo 4. NVIDIA Container Toolkit已正确安装
        echo 5. GPU驱动程序已安装
    )
)

pause
