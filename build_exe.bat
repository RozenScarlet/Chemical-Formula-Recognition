@echo off
chcp 65001 > nul
title 化学方程式OCR推理工具打包器

echo.
echo ================================================================
echo                    化学方程式OCR推理工具打包器
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

REM 运行打包脚本
echo.
echo 开始打包过程...
python build_exe.py

echo.
echo 打包完成！
pause