import os

def fiftyone_start():
    root = os.path.split(os.path.realpath(__file__))[0]
    # 环境变量设置配置文件地址
    os.environ["FIFTYONE_CONFIG_PATH"] = os.path.join(root,"fo_config.json")

    import fiftyone as fo
    print("FiftyOne Server Running...")
    session = fo.launch_app(desktop=False,auto=False) 

    print("FiftyOne Server Startup Complete!")
    # session.wait()

# fiftyone_start()