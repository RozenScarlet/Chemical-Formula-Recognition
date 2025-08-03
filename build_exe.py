"""
使用PyInstaller打包推理工具为exe文件的脚本
生成一个独立的exe文件，包含所有依赖项
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_pyinstaller():
    """检查PyInstaller是否安装"""
    try:
        import PyInstaller
        print(f"✓ PyInstaller已安装 (版本: {PyInstaller.__version__})")
        return True
    except ImportError:
        print("✗ PyInstaller未安装")
        return False

def install_pyinstaller():
    """安装PyInstaller"""
    print("正在安装PyInstaller...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("✓ PyInstaller安装成功")
        return True
    except subprocess.CalledProcessError:
        print("✗ PyInstaller安装失败")
        return False

def create_spec_file():
    """创建PyInstaller spec文件"""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# 数据文件和隐藏导入
datas = [
    ('src', 'src'),
]

hiddenimports = [
    'torch',
    'torchvision',
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    'numpy',
    'cv2',
    'sklearn',
    'sklearn.model_selection',
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.font_manager',
    'tqdm',
    'psutil',
    'transformer_cnn_net',
    'dataset',
    'utils',
]

# 分析脚本
a = Analysis(
    ['inference_tool.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# 收集所有文件
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# 生成可执行文件
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='化学方程式OCR推理工具',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # 不显示控制台窗口
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if os.path.exists('icon.ico') else None,
)
'''
    
    with open('inference_tool.spec', 'w', encoding='utf-8') as f:
        f.write(spec_content)
    print("✓ spec文件创建成功")

def build_exe():
    """构建exe文件"""
    print("开始构建exe文件...")
    print("这可能需要几分钟时间，请耐心等待...")
    
    try:
        # 使用spec文件构建
        subprocess.check_call([
            'pyinstaller', 
            '--clean',
            '--noconfirm',
            'inference_tool.spec'
        ])
        print("✓ exe文件构建成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ exe文件构建失败: {e}")
        return False

def copy_model_files():
    """复制模型文件到dist目录"""
    dist_dir = Path('dist')
    if not dist_dir.exists():
        print("✗ dist目录不存在")
        return False
        
    # 查找exe文件目录
    exe_dir = None
    for item in dist_dir.iterdir():
        if item.is_dir():
            exe_dir = item
            break
            
    if exe_dir is None:
        print("✗ 未找到exe目录")
        return False
        
    print(f"复制文件到: {exe_dir}")
    
    # 复制模型文件（如果存在）
    model_files = [
        'checkpoints',
        'best_model.pth',
        'model.pth'
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            if os.path.isdir(model_file):
                shutil.copytree(model_file, exe_dir / model_file, dirs_exist_ok=True)
                print(f"✓ 复制目录: {model_file}")
            else:
                shutil.copy2(model_file, exe_dir / model_file)
                print(f"✓ 复制文件: {model_file}")
    
    # 创建README文件
    readme_content = """# 化学方程式OCR推理工具

## 使用方法
1. 将需要识别的图片文件放在exe文件同目录下
2. 双击运行"化学方程式OCR推理工具.exe"
3. 点击"开始推理当前目录图片"按钮
4. 等待推理完成，查看结果
5. 可以点击"保存结果"保存识别结果到文件

## 支持的图片格式
- JPG/JPEG
- PNG  
- BMP
- TIFF/TIF
- GIF

## 注意事项
- 确保图片清晰，化学方程式完整
- 推理速度取决于图片数量和计算机性能
- 结果会自动保存为JSON和TXT格式

## 技术支持
如有问题请联系开发团队
"""
    
    with open(exe_dir / 'README.txt', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("✓ 创建README文件")
    
    return True

def cleanup():
    """清理临时文件"""
    cleanup_items = [
        'build',
        '__pycache__',
        'inference_tool.spec'
    ]
    
    for item in cleanup_items:
        if os.path.exists(item):
            if os.path.isdir(item):
                shutil.rmtree(item)
                print(f"✓ 删除目录: {item}")
            else:
                os.remove(item)
                print(f"✓ 删除文件: {item}")

def main():
    """主函数"""
    print("=" * 60)
    print("化学方程式OCR推理工具 - 打包脚本")
    print("=" * 60)
    
    # 检查当前目录
    if not os.path.exists('inference_tool.py'):
        print("✗ 未找到inference_tool.py文件")
        print("请确保在正确的目录下运行此脚本")
        return False
        
    # 检查PyInstaller
    if not check_pyinstaller():
        if not install_pyinstaller():
            return False
            
    # 创建spec文件
    create_spec_file()
    
    # 构建exe
    if not build_exe():
        return False
        
    # 复制模型文件
    if not copy_model_files():
        print("⚠ 模型文件复制可能不完整，请手动复制模型文件到exe目录")
        
    # 清理临时文件
    cleanup()
    
    print("=" * 60)
    print("✓ 打包完成!")
    print("exe文件位置: dist/化学方程式OCR推理工具/")
    print("可以将整个目录复制到其他电脑使用")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    input("\\n按回车键退出...")
    sys.exit(0 if success else 1)