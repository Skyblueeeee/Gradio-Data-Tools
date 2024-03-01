import os, json

def print_statistics(root_dir):
    categoryStatusDict = {}
    totalBoxes = 0
    totalImgs = 0
    usageImgSet = set()
    for group in os.walk(root_dir):
        for each_file in group[2]:
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
                for eachCatID in catDict:
                    catName = catDict[eachCatID]
                    tempIndex = categoryStatusDict.get(catName, [catName, 0, set()])
                    categoryStatusDict[catName] = tempIndex

                for eachAnnotation in annotations:
                    catName = catDict[eachAnnotation["category_id"]]
                    tempIndex = categoryStatusDict.get(catName, [catName, 0, set()])
                    tempIndex[1] += 1
                    tempIndex[2].add(imgDict[eachAnnotation["image_id"]])

                    categoryStatusDict[catName] = tempIndex

    res = []
    for each in categoryStatusDict.keys():
        res.append ([
                        categoryStatusDict[each][0], 
                        categoryStatusDict[each][1], 
                        len(categoryStatusDict[each][2]), 
                        round(categoryStatusDict[each][1]/(len(categoryStatusDict[each][2]) + 1e-8),2)] )
        usageImgSet = usageImgSet | categoryStatusDict[each][2]
        totalBoxes += categoryStatusDict[each][1]
    res.append(["图片总数量",totalImgs,"总框数", totalBoxes])
    res.append(["图片利用数",len(usageImgSet),"图片利用率", str(round(len(usageImgSet)*100/(totalImgs + 1e-8),2))+"%"])

 
    return res

# print(len(print_statistics(r"Y:\label\label_P-IJC23110092_岚图总装检测\车门工位\标注图\MJSB标签取出")))