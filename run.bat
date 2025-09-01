@echo off
echo 运行Docker容器...
echo 将src目录映射到容器的/app目录

set CURRENT_DIR=%cd%
echo 当前目录: %CURRENT_DIR%
echo 源代码目录: %CURRENT_DIR%\src

docker run -it --rm ^
    -v "%CURRENT_DIR%\src:/app/src" ^
    --name llm-0-1-container ^
    llm-0-1:latest

if %ERRORLEVEL% NEQ 0 (
    echo 容器运行失败！
    echo 请确保：
    echo 1. Docker正在运行
    echo 2. 镜像llm-0-1:latest已构建
    echo 3. src目录存在
)

pause
