from PyInstaller.__main__ import run
import shutil
import os

# -F:打包成一个EXE文件
# -w:不带console输出控制台，window窗体格式
# --paths：依赖包路径
# --icon：图标
# --noupx：不用upx压缩
# --clean：在构建之前清理PyInstaller缓存并删除临时文件

FILE_PATH, FILE_FULL_NAME = os.path.split(os.path.realpath(__file__))
NEW_PATH = FILE_PATH + '/dist/main'

if __name__ == '__main__':
    if os.path.exists(FILE_PATH + '/build'):
        shutil.rmtree(FILE_PATH + '/build')
    if os.path.exists(FILE_PATH + '/dist'):
        shutil.rmtree(FILE_PATH + '/dist')

    opts = ['-D',
            '-w',
            '--icon=./graphics/icon.ico',
            'main.py']
    run(opts)

    shutil.copy(FILE_PATH + '/about.ui', NEW_PATH)
    shutil.copytree(FILE_PATH + '/panels', NEW_PATH + '/panels')
    shutil.copytree(FILE_PATH + '/doc', NEW_PATH + '/doc')
    shutil.copytree(FILE_PATH + '/graphics', NEW_PATH + '/graphics')
    os.renames(NEW_PATH + "/main.exe", NEW_PATH + "/APP.exe")
