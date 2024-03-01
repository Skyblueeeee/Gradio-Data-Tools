import os
import openpyxl
import time
from datetime import datetime, timedelta

from ...configs.dir_config import config_dir

def file_date():
    date_list = []
    origin_image = config_dir["project"]
    for date_dir in os.listdir(origin_image):
        date_list.append(date_dir.replace("-",""))
    return date_list

def biaozhutu(biaozhutu_dir,start,end):
    if len(os.listdir(biaozhutu_dir))==0:
        print("请补充数据")
    else:
        time_dict = {}
        print("标注图数量:")
        print("     ","发图时间","        ","图数量")
        for each_date_dir in os.listdir(biaozhutu_dir):
            if each_date_dir == None:
                print(1)
            else:
                jpg_list = []
                if "发图时间" in each_date_dir and "." not in each_date_dir:
                    if int(each_date_dir[5:]) >= int(start[5:]) and int(each_date_dir[5:]) <= int(end[5:]):
                        for each_group in os.walk(os.path.join(biaozhutu_dir,each_date_dir)):
                            for each_jpg in each_group[2]:
                                if each_jpg.endswith((".jpg",".bmp",".png")):
                                    jpg_list.append(each_jpg)
                        print("     ",each_date_dir,' ',len(jpg_list))

                        time_dict[each_date_dir] = [len(jpg_list),""]

def yuantu(yuantu_dir,start,end):
    # 将字符串转换为 datetime 对象
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")

    # 生成时间列表
    date_list = []
    while start_date <= end_date:
        date_list.append(start_date.strftime("%Y%m%d"))
        start_date += timedelta(days=1)    
    if len(os.listdir(yuantu_dir))==0:
        print("请补充数据")
    else:
        time = {}
        
        print("原图:")
        print("       ","原图时间","车型","车数量","图数量")

        for date in os.listdir(yuantu_dir):
            jpg_list = []
            type_list = []
            each_group_list = []
            if date.replace("-","") in date_list:
                for each_dir in os.listdir(os.path.join(yuantu_dir,date)):
                    car_type = each_dir.split("_")[0]
                    new_dir = os.path.join(yuantu_dir, date, each_dir)
                    if car_type not in type_list:
                        type_list.append(car_type)
                    for each_group in os.listdir(new_dir):
                        each_group_list.append(each_group)
                    for each_group in os.walk(new_dir):
                        for each_jpg in each_group[2]:
                            if each_jpg.endswith(("1.jpg",".bmp",".png")):
                                jpg_list.append(each_jpg)
                    time[each_dir] = [len(each_group_list),len(jpg_list)]
                print("     ",date," ",len(car_type)," ",len(each_group_list)," ",len(jpg_list)) 
