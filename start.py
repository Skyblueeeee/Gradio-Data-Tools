import subprocess
from log.log import log
import logging
from datetime import datetime
import time
import sys

sys.argv[-1]

python38_path = sys.argv[-1]
main_script_path = r'main.py'
log_file_path = "log/gdt_log.log"

# 日志服务
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename=log_file_path, filemode='a', force=True)
logging.info('Gradio-Data-Tools')
logging.info('Starting Gradio-Data-Tools Server...')
logging.info("Point your web browser to http://10.10.1.129:8001/")
logging.info('G-D-T Server：Fiftyone Initializing the database...')
main_process = subprocess.Popen([python38_path, main_script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def check_process(process):
    return process.poll() is None

def print_output(process, prefix):
    while check_process(process):
        output = process.stdout.readline().decode(sys.stdout.encoding, errors='replace').strip()
        if output:
            logging.info('%s: %s', prefix, output)
        time.sleep(0.1)

import threading
main_thread = threading.Thread(target=print_output, args=(main_process, 'G-D-T Server'))

main_thread.start()
main_thread.join()
main_process.wait()

if main_process.returncode == 0:
    logging.info('Gradio-Data-Tools Server Started Successfully!')
else:
    logging.error('Gradio-Data-Tools Server Failed To Start!')
