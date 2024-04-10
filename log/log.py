import logging
import time
import os
import sys

def flow_api(CF_fun):
    def wrapper(self, *args, **kwargs):
        tick = time.time()
        try:
            result = CF_fun(self, *args, **kwargs)
        except Exception as e:
            tock = time.time()
            print("Gradio-Data-Tools Running: " + CF_fun.__name__ + f" Finished Time {(tock - tick):.2f}s", end="\n\n")
            raise e
        else:
            tock = time.time()
            print("Gradio-Data-Tools Running: " + CF_fun.__name__ + f" Finished Time {(tock - tick):.2f}s", end="\n\n")
            return result
    return wrapper

def timeit(self: logging.Logger, period_name, tick):
    if self.level <= logging.INFO and tick != None:
        tock = time.time()
        self.info(f"{period_name}耗时为：{(tock - tick)*1000:>8.2f}ms")
        return tock
    else:
        return None

def begin_time(self: logging.Logger):
    if self.level <= logging.INFO:
        return time.time()
    else:
        return None

def get_logger(model_cfg: dict, log_level):
    setattr(logging.Logger, "begin_time", begin_time)
    setattr(logging.Logger, "timeit", timeit)
    logger = logging.getLogger(model_cfg["model_name"])

    if model_cfg.get("log_level", "") == "":
        model_cfg["log_level"] = log_level
    
    log_level = eval("logging." + model_cfg["log_level"].upper())
    if log_level <= logging.DEBUG:
        logger.setLevel(logging.DEBUG)
        h = logging.StreamHandler(sys.stdout)
        h.formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        logger.addHandler(h)
    else:
        logger.setLevel(log_level)
        dir_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "once_logs")
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        file_name = os.path.join(dir_name, model_cfg["model_name"] + ".log")
        h = logging.FileHandler(file_name, "w", encoding="utf-8")
        h.formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        logger.addHandler(h)
    return logger