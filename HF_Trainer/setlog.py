import os
import logging
import transformers
import datasets
import pynvml

from config import trainer_args


def create_logger():
    """
    1. 获取 Logger 对象的方法为 getLogger(), 单例模式, 整个系统只有一个 root Logger 对象
    2. Logger 对象可以设置多个 Handler 对象, Handler 对象又可以设置 Formatter 对象
        针对有时候我们既想在控制台中输出DEBUG 级别的日志，又想在文件中输出WARNING级别的日志。
        可以只设置一个最低级别的 Logger 对象，两个不同级别的 Handler 对象。
    3. 临时禁止输出日志，logger.disabled = True

    """
    # 1. 创建logger
    logger = logging.getLogger("logger")

    # 2. 创建handler
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(
        filename="./output/log_manual.log", mode="w")
    if not os.path.exists("./output"):
        os.makedirs("./output")

    # 3. 设置输出等级
    logger.setLevel(logging.DEBUG)  # 通常设置为两个handler输出中的最低等级
    handler1.setLevel(logging.WARNING)
    handler2.setLevel(logging.DEBUG)

    # 4. 设置输出格式
    formatter = logging.Formatter(fmt="[%(asctime)s] [%(levelname)s] [manual] %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)

    # 5. 添加到logger
    logger.addHandler(handler1)
    logger.addHandler(handler2)

    return logger


def print_gpu_utilization(index):
    """
    显示第index个GPU的显存使用情况
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization(0)


def setlog():
    logger = create_logger()
    # set the main code and the modules it uses to the same log-level according to the node
    log_level = trainer_args["log_level"]
    level = {
        "debug": 10,
        "info": 20,
        "warning": 30,
        "error": 40,
        "critical": 50
    }
    log_level_int = level[log_level]
    logger.setLevel(log_level_int)
    datasets.utils.logging.set_verbosity(log_level_int)
    transformers.utils.logging.set_verbosity(log_level_int)

    logger.debug("set the logger debug")
    logger.warning("set the logger warning")

    return logger
