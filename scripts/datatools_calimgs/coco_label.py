import os, json
import pandas as pd
import gradio as gr

def plot(a):
    new_dict = {}
    x = []
    y = []
    for i in a[:-2]:
        x.append(i[0])
        y.append(i[1])
        new_dict["x"] = x
        new_dict["y"] = y
    return gr.ScatterPlot.update(value=pd.DataFrame(new_dict),x="x",y="y",title="标签散点图",
                                 x_title="Label",y_title="Number",height=400,width=1400),gr.LinePlot.update(value=pd.DataFrame(new_dict),
                                 x="x",y="y",title="标签折线图",x_title="Label",y_title="Number",height=400,width=1400),gr.BarPlot.update(value=pd.DataFrame(new_dict),
                                 x="x",y="y",title="标签条形图",x_title="Label",y_title="Number",height=400,width=1400)

def print_statistics(root_dir):
    categoryStatusDict = {}
    totalBoxes = 0
    totalImgs = 0
    usageImgSet = set()
    for group in os.walk(root_dir):
        for each_file in group[2]:
            new_dict = {}
            if "coco" in each_file and each_file.endswith(".json"):
                json_path = os.path.join(group[0], each_file)
                with open(json_path, "r", encoding="utf-8") as fp:
                    cocoDict = json.load(fp)

                images = cocoDict.pop("images")
                categories = cocoDict.pop("categories")
                annotations = cocoDict.pop("annotations")

                # 总图片数量
                totalImgs += len(images)
                
                # 统计每个标签的框出数
                catDict = {each["id"]: each["name"] for each in categories}
                imgDict = {each["id"]: each["file_name"] for each in images}
                for eachCatID,eachlabel in catDict.items():
                    catName = catDict[eachCatID]
                    tempIndex = categoryStatusDict.get(catName, [catName, 0, set()])
                    categoryStatusDict[catName] = tempIndex
                    new_dict[eachlabel] = ""

                for eachAnnotation in annotations:
                    catName = catDict[eachAnnotation["category_id"]]
                    tempIndex = categoryStatusDict.get(catName, [catName, 0, set()])
                    tempIndex[1] += 1
                    if isinstance(eachAnnotation["image_id"],str):
                        imgs_id = int(eachAnnotation["image_id"])
                        tempIndex[2].add(imgDict[imgs_id])
                    else:
                        tempIndex[2].add(eachAnnotation["image_id"])

                    categoryStatusDict[catName] = tempIndex

            # if "via_" in each_file and each_file.endswith(".json"):
            #     json_path = os.path.join(group[0], each_file)
            #     with open(json_path, "r", encoding="utf-8") as fp:
            #         viaDict = json.load(fp)
            #         label_dict = viaDict["_via_attributes"]["region"]
            #         for keys in label_dict.items():
            #             n_dict = {v: k for k, v in keys[1]["options"].items()}
            #         new_dict = n_dict        
    res = []
    for each in categoryStatusDict.keys():
        res.append ([
                        # new_dict[categoryStatusDict[each][0]],
                        categoryStatusDict[each][0], 
                        categoryStatusDict[each][1], 
                        len(categoryStatusDict[each][2]), 
                        round(categoryStatusDict[each][1]/(len(categoryStatusDict[each][2]) + 1e-8),2)] )
        usageImgSet = usageImgSet | categoryStatusDict[each][2]
        totalBoxes += categoryStatusDict[each][1]
    res.append(["图片总数量",totalImgs,"总框数", totalBoxes])
    res.append(["图片利用数",len(usageImgSet),"图片利用率", str(round(len(usageImgSet)*100/(totalImgs + 1e-8),2))+"%"])
    # print(res)

    label_scatter_plot,label_line_plot,label_bar_plot = plot(res)

    return res,len(usageImgSet),label_scatter_plot,label_line_plot,label_bar_plot

# print(print_statistics(r"\\10.10.1.125\ai01\label\label_P-IJC23110092_岚图总装检测\仪表工位\标注图\发图时间_240115\HEC"))