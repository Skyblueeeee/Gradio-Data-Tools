import os
import re
import sys
import json
import time

root = os.path.split(os.path.realpath(__file__))[0]
config_abspath = os.path.join(root[:-13],"fo_config.json")
# print(config_abspath)

if sys.argv[-1].endswith(".json"):
    config_abspath = sys.argv[-1]

def parse_json(config_abspath):
    with open(config_abspath, "r") as fp:
        config = json.load(fp)
        database_uri = config["database_uri"]
        database_dir = config["fo_data"]
        m = re.search(r"/((\d|[1-9]\d|1\d{2}|2[0-4]\d|25[0-5])\.(\d|[1-9]\d|1\d{2}|2[0-4]\d|25[0-5])\.(\d|[1-9]\d|1\d{2}|2[0-4]\d|25[0-5])\.(\d|[1-9]\d|1\d{2}|2[0-4]\d|25[0-5])\:([0-9]|[1-9]\d{1,3}|[1-5]\d{4}|6[0-4]\d{3}|65[0-4]\d{2}|655[0-2]\d|6553[0-5]))/", database_uri)
        bind_ip, port = m.group(1).split(":")
        dbpath = os.path.realpath(database_dir)
        os.makedirs(dbpath, exist_ok=True)
        logpath = os.path.join(dbpath, "log", "mongo.log")
        os.makedirs(os.path.dirname(logpath), exist_ok=True)
    return bind_ip, port, logpath

def mongo_server(config_abspath):
    bind_ip, port, logpath = parse_json(config_abspath)

    os.system(
        " ".join(
            [
                '--logpath',
                logpath,
                '--port',
                port,
                '--bind_ip',
                bind_ip
                # '--nounixsocket'
            ]
        )
    )

def app_server(config_abspath):
    os.environ["FIFTYONE_CONFIG_PATH"] = config_abspath

    import fiftyone as fo

    session = fo.launch_app()
    while True:
        time.sleep(300)

def main():
    global config_abspath
    from multiprocessing import Process
    p = Process(target=mongo_server, args=(config_abspath, ))
    p.daemon = True
    p.start()
    time.sleep(10)
    # p = Process(target=app_server, args=(config_abspath, ))
    # p.daemon = True
    # p.start()
    while True:
        time.sleep(300)

if __name__ == "__main__":
    main()

