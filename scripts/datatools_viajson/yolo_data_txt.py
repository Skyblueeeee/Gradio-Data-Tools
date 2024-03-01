import os

date_list = [
        20220728,20220729,20220819
]

# 多文件夹数据地址
data_path = r""
# 替换地址得前缀地址
root_path = r"/root/datayard/POC/ann_img"
# 新建train.txt地址
txt_path = r""
# 移除残留得txt文件
if os.path.exists(txt_path):
    os.remove(txt_path)

for date in date_list:
    date = str(date)
    data_ = data_path + "/"+ date
    root_ = root_path + "/" + date

    for file_name in os.walk(data_):
        for file in file_name[2]:
            if file.endswith("jpg"):
                with open(txt_path,"a",encoding="utf-8") as data:
                    data.write(root_ + "/" +file +"\n")
                    data.close()

    print(f"{date}文件夹写入完成")