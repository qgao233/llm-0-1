@echo off
echo 强制使用CPU模式运行Docker容器...
echo 将src目录映射到容器的/app目录

set CURRENT_DIR=%cd%
echo 当前目录: %CURRENT_DIR%
echo 源代码目录: %CURRENT_DIR%\src

echo 启动CPU容器...
docker run -it --rm ^
    -v "%CURRENT_DIR%\src:/app/src" ^
    --name llm-0-1-cpu ^
    llm-0-1:latest

if %ERRORLEVEL% NEQ 0 (
    echo 容器运行失败！
    echo 请确保：
    echo 1. Docker正在运行
    echo 2. 镜像llm-0-1:latest已构建
    echo 3. src目录存在
    goto :error
)

echo 容器已成功退出
goto :end

:error
echo 运行失败，请检查上述要求
pause
exit /b 1

:end
pause
