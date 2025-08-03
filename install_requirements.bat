@echo off
chcp 65001 > nul
title 安装打包依赖

echo.
echo ================================================================
echo                    安装打包所需依赖包
echo ================================================================
echo.

REM 检查Python环境
python --version > nul 2>&1
if errorlevel 1 (
    echo ✗ Python未安装或未添加到PATH环境变量
    echo 请先安装Python 3.8+并添加到环境变量
    pause
    exit /b 1
)

echo ✓ Python环境检查通过

REM 升级pip
echo.
echo 升级pip...
python -m pip install --upgrade pip

REM 安装依赖
echo.
echo 安装依赖包...
python -m pip install -r requirements_exe.txt

echo.
echo ✓ 依赖包安装完成！
echo 现在可以运行 build_exe.bat 进行打包
pause