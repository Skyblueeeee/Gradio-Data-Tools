import os
import json
from shutil import copy
from scripts.datatools_viajson.via2coco import run_viatococo

def label_pos(root_path,model,jsonname,tagparentname):
    if os.path.exists(os.path.join(root_path,jsonname)):
        with open(os.path.join(root_path, jsonname), "r", encoding="utf-8") as fp:
            cocoDict = json.load(fp)
        categoriesDict = cocoDict["_via_attributes"]["region"][tagparentname]["options"]

        tips = f"当前 {model} 有的标签:\n"
        for eachKey in categoriesDict.keys():
            tips += (eachKey + "    ")

        print(tips)
    else:
        print("\n路径有误或无法解析: {}".format(root_path))


def run_quqian(root_path,jsonname1,tagparentname,save_path,label_name,pos,iscoco,issuper,iscopys,img_width,img_heigh):
    jsonname = jsonname1 + ".json"
    SAVENAME = "via_gr_getimgs"
    label_dict ={}
    if os.path.exists(os.path.join(root_path, jsonname)):
        with open(os.path.join(root_path, jsonname), "r", encoding="utf-8") as fp:
            cocoDict = json.load(fp)
    
        categoriesDict = cocoDict["_via_attributes"]["region"][tagparentname]["options"]
        remainList = label_name.split(" ")
        for each in remainList:
            if each in categoriesDict.keys():
                imgDict = cocoDict["_via_img_metadata"]
                newImgDict = {}
                newImgPath = []

                tempKeyList = [each for each in imgDict.keys()]
                for eachImg in tempKeyList:
                    newregions = []
                    try:
                        tempDict = imgDict[eachImg]
                        toPop = False
                        if pos == False:
                            for eachRegion in tempDict["regions"]:
                                if eachRegion["region_attributes"][tagparentname] in remainList:
                                    newImgPath.append(tempDict["filename"])
                                    newregions.append(eachRegion)
                                    toPop = True
                                    continue
                            if toPop:
                                tempDict["regions"] = newregions
                                tempDict = imgDict.pop(eachImg)
                                newImgDict[eachImg] = tempDict
                        else:
                            for eachRegion in tempDict["regions"]:
                                if eachRegion["region_attributes"][tagparentname] in remainList:
                                    newImgPath.append(tempDict["filename"])
                                    toPop = True
                                    break
                            if toPop:
                                tempDict = imgDict.pop(eachImg)
                                newImgDict[eachImg] = tempDict

                    except Exception as e:
                        return "异常",str(e)
                # exit()

                cocoDict["_via_img_metadata"] = newImgDict

                if issuper:
                    pass
                else:
                    label_dict[each] = cocoDict["_via_attributes"]["region"][tagparentname]["options"][each]
                    cocoDict["_via_attributes"]["region"][tagparentname]["options"] = label_dict
                    cocoDict["_via_attributes"]["region"][tagparentname]["default_options"] = {}

                newDir = os.path.join(save_path, "_".join(remainList) + "标签取出")
                if os.path.exists(newDir):
                    return "该文件夹已存在, 图片可能覆盖!"
                else:
                    os.makedirs(newDir)

                with open(os.path.join(newDir, SAVENAME+".json"), "w", encoding="utf-8") as fp:
                    json.dump(cocoDict, fp,ensure_ascii=False)
                
                # 是否复制图片
                if iscopys:
                    for eachPath in newImgPath:
                        copy(os.path.join(root_path, eachPath), os.path.join(newDir, eachPath))
                # 调用转coco函数
                if iscoco == True:
                    run_viatococo(newDir,SAVENAME, "coco_gr", tagparentname,img_width,img_heigh)
                return "取图任务已完成!"
            else:
                return "标签不在该文件中!!!\n"
    else:
        return "\n路径有误或无法解析: {}".format(os.path.join(root_path, jsonname))

# root_path = r"\\10.10.1.125\ai01\codeyard\my_codeyard\data\train\GA\test"
# save_path = r"D:\shy_code\test1"
# jsonname1="via_project"
# tagparentname = "fitow"
# label_name = "qr"
# pos = False
# iscoco = False
# issuper = False
# iscopys = False
# print(run_quqian(root_path,jsonname1,tagparentname,save_path,label_name,pos,iscoco,issuper,iscopys))