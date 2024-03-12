import os

dir = r"Z:\P-FTY221009-01_花都宝井焊缝检测项目\POC\ann_img\20220819"

for i in os.listdir(dir):
    if i.endswith("jpg"):
        print("images/"+i)