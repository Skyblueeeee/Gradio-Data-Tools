import subprocess
import logging
from datetime import datetime
import time
import sys
import threading

# 获取命令行参数中的python38路径
python38_path = sys.argv[-1]
main_script_path = r'main.py'
log_file_path = "log/gdt_log.log"

# 完整配置日志服务，包括将日志写入文件
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=log_file_path,
                    filemode='a',
                    force=True)

# 输出控制台日志
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger('').addHandler(console_handler)

logging.info('Gradio-Data-Tools')
logging.info('Starting Gradio-Data-Tools Server...')
logging.info("Please point your web browser to http://10.10.1.129:8001/")
logging.info('G-D-T Server: Initializing Fiftyone database...')

# 启动主进程
main_process = subprocess.Popen([python38_path, main_script_path],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT) 

def check_process(process):
    return process.poll() is None

def print_output(process, prefix):
    while check_process(process):
        output = process.stdout.readline().decode(sys.stdout.encoding, errors='replace').strip()
        if output:
            logging.info('%s: %s', prefix, output.strip())
        time.sleep(0.1)

# 创建并启动用于输出子进程信息的新线程
main_thread = threading.Thread(target=print_output, args=(main_process, 'G-D-T Server'))
main_thread.start()

# 等待主线程完成
main_thread.join()

# 检查主进程返回码
if main_process.returncode == 0:
    logging.info('Gradio-Data-Tools Server Started Successfully!')
else:
    logging.error('Gradio-Data-Tools Server Failed To Start!')

# 清理：移除控制台处理器（可选，取决于是否需要持续显示控制台日志）
logging.getLogger('').removeHandler(console_handler)