import os
import cv2
import json
import imgaug as ia
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from threading import Thread, Lock



cocoJsonName = "coco"
N_THREADS = 12 # 多线程数目
SEED = 123456 # 随机种子, 可以为None

"""README
1.目录结构:
执行前:
ROOTDIR__
        |__imageDir1__cocoJsonName.json
        |           |
        |           |__images
        |
        |__imageDir2__cocoJsonName.json
        |           |
        |           |__images
        |
        |_等等
执行后:
ROOTDIR__
        |__imageDir1__cocoJsonName.json
        |           |
        |           |__images
        |
        |__imageDir1_aug__cocoJsonName.json
        |               |
        |               |__images
        |
        |__imageDir2__cocoJsonName.json
        |           |
        |           |__images
        |
        |__imageDir2_aug__cocoJsonName.json
        |               |
        |               |__images
        |
        |_等等

2.自定义:

"""

GlobalAugDict = {}
GlobalAugNumDict = {}
threadLock = Lock()

def getAugMethod(dirName):
    for eachKey in GlobalAugDict.keys():
        if eachKey in dirName:
            return eachKey
    return "global_default_aug_method"

def getAugNum(dirName):
    for eachKey in GlobalAugNumDict.keys():
        if eachKey in dirName:
            return GlobalAugNumDict[eachKey]
    return GlobalAugNumDict["global_default_aug_number"]

def check(ROOTDIR,check_pos):
    # 目录检查和创建
    workImageDirs = []
    augImageDirs = []
    i = 0
    T = True
    for eachImageDir in os.listdir(ROOTDIR):
        if os.path.isdir(os.path.join(ROOTDIR, eachImageDir)):
            if (cocoJsonName + ".json") in os.listdir(os.path.join(ROOTDIR, eachImageDir)):
                workImageDirs.append(os.path.join(ROOTDIR, eachImageDir))
                augImageDir = os.path.join(ROOTDIR, eachImageDir + "_aug")
                if os.path.exists(augImageDir):
                    if (cocoJsonName + ".json") in os.listdir(augImageDir):
                        print("数据安全警告: {}目录已存在{}, 数据增强过程中会覆盖该文件".format(eachImageDir + "_aug", cocoJsonName + ".json"))
                        T = False
                        
                else:
                    augImageDirs.append(augImageDir)
                i += 1
                if check_pos == True:
                    return False

    if T == True:
        print("此次数据增强将应用到以下{}个目录: ".format(i))
        for eachWorkImageDir in workImageDirs:
            tempName = os.path.basename(eachWorkImageDir)
            print("{}:\t增强模式: {}\t增强次数: {}".format(tempName, getAugMethod(tempName), getAugNum(tempName)))

        for eachAugImageDir in augImageDirs:
            os.mkdir(eachAugImageDir)
        return workImageDirs

def image_dir_aug(ROOTDIR,x1,x2,y1,y2,ml1,ml2,count,display,flip):
    #############################################################################
    # 注册 根据目录名使用的数据增强方法
    # imgaug官方文档
    # https://imgaug.readthedocs.io/en/latest/source/examples_bounding_boxes.html

    GlobalAugDict["global_default_aug_method"] = [
        iaa.Sequential([
            # iaa.OneOf([
            #     iaa.Affine(translate_px={"x":300}),
            #     iaa.Affine(translate_px={"x":-300}),
            # ])
            iaa.Fliplr(flip),
            iaa.Affine(
                translate_percent={"x":(x1, x2), "y":(y1, y2)},
            ),
            iaa.Multiply((ml1, ml2))
        ])
    ]
    # GlobalAugDict["Z"] = [
    #     iaa.Affine(translate_percent={"x":(-0.03, 0.03)})
    # ]
    # GlobalAugDict["RD260"] = [
    #     iaa.Affine(translate_percent={"x":(-0.03, 0.03), "y":(-0.03, 0.03)})
    # ]
    # GlobalAugDict["LD"] = [
    #     iaa.Affine(translate_percent={"x":(-0.03, 0.03), "y":(-0.03, 0.03)})
    # ]
    # GlobalAugDict["RD"] = [
    #     iaa.Affine(translate_percent={"x":(-0.03, 0), "y":(-0.03, 0.03)})
    # ]
    # GlobalAugDict["LT"] = [
    #     iaa.Affine(translate_percent={"x":(-0.03, 0.03), "y":(-0.03, 0.03)})
    # ]
    # GlobalAugDict["RT"] = [
    #     iaa.Affine(translate_percent={"x":(-0.03, 0.03), "y":(-0.03, 0.03)})
    # ]
    #############################################################################
    # 注册 根据目录名, 每个目录每张图片的重复增强次数
    GlobalAugNumDict["global_default_aug_number"] = count
    # GlobalAugNumDict["RD260"] = 3
    # GlobalAugNumDict["D"] = 5
    # GlobalAugNumDict["T"] = 4
    # GlobalAugNumDict["Z"] = 2
    #############################################################################

    def run(images):
        global threadLock
        nonlocal IMAGEID, ANNOTATIONID, aug_method, aug_number
        nonlocal eachWorkImageDir, annotationsDict, tarCOCODict, img_aug
        seq = iaa.Sequential(aug_method)
        for eachImage in images:
            imageName = eachImage["file_name"]
            imagePath = os.path.join(eachWorkImageDir, imageName)
            image_id = eachImage["id"]
            try:
                image = cv2.imdecode(np.fromfile(imagePath, dtype=np.uint8), cv2.IMREAD_COLOR)

            except Exception as e:
                print("{}图片读取错误: {}".format(imagePath, str(e)))
                continue

            try:
                tempBoxes = []
                for eachAnnotation in annotationsDict.get(image_id, []):
                    x1, y1, w, h = eachAnnotation["bbox"]
                    tempBox = BoundingBox(x1, y1, x1 + w, y1 + h)
                    tempBox.label = (eachAnnotation["category_id"], eachAnnotation["iscrowd"])
                    tempBoxes.append(tempBox)
                bbs = BoundingBoxesOnImage(tempBoxes, shape=image.shape)

                p, s = os.path.splitext(imageName)
                for i in range(aug_number):
                    eachImage["file_name"] = p + '_' + str(i) + s
                    # import random
                    # eachImage["file_name"] =  str(random.randint(1000, 1000000000)) + s
                    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
                    img = cv2.imencode(s, image_aug)[1].tofile(os.path.join(eachWorkImageDir+"_aug", eachImage["file_name"]))
                    img_aug.append(image_aug)
                    threadLock.acquire()
                    eachImage["id"] = IMAGEID
                    IMAGEID += 1
                    tarCOCODict["images"].append(eachImage.copy())
                    threadLock.release()

                    for eachBox in bbs_aug.bounding_boxes:
                        eachAnnotation["image_id"] = eachImage["id"]
                        eachAnnotation["category_id"] = eachBox.label[0]
                        if "numpy" in str(type(eachBox.x1)):
                            x1, y1, x2, y2 = eachBox.x1.item(), eachBox.y1.item(), eachBox.x2.item(), eachBox.y2.item()
                        else:
                            x1, y1, x2, y2 = eachBox.x1, eachBox.y1, eachBox.x2, eachBox.y2
                        x, y, w, h = x1, y1, x2 - x1, y2 - y1
                        eachAnnotation["bbox"] = [x, y, w, h]
                        eachAnnotation["segmentation"] = [x, y, x + w, y, x + w, y + h, x, y + h]
                        eachAnnotation["iscrowd"] = eachBox.label[1]

                        threadLock.acquire()
                        eachAnnotation["id"] = ANNOTATIONID
                        ANNOTATIONID += 1
                        tarCOCODict["annotations"].append(eachAnnotation.copy())
                        threadLock.release()
            except Exception as e:
                print("{}图片可能无框: {}".format(imagePath, str(e)))
                continue
        
    if SEED != None:
        ia.seed(SEED)
    workImageDirs = check(ROOTDIR,False)
    img_aug = []
    if workImageDirs != False:
        for eachWorkImageDir in workImageDirs:
            srcCOCOPath = os.path.join(eachWorkImageDir, cocoJsonName + ".json")
            tarCOCOPath = os.path.join(eachWorkImageDir + "_aug", cocoJsonName + ".json")
            with open(srcCOCOPath, "r", encoding="utf-8") as fp:
                srcCOCODict = json.load(fp)

            tarCOCODict = {}
            tarCOCODict["licenses"] = srcCOCODict.pop("licenses")
            tarCOCODict["categories"] = srcCOCODict.pop("categories")
            tarCOCODict["info"] = srcCOCODict.pop("info")
            tarCOCODict["images"] = []
            tarCOCODict["annotations"] = []

            annotationsDict = {}
            for annotation in srcCOCODict['annotations']:
                image_id = int(annotation['image_id'])
                if image_id not in annotationsDict:
                    annotationsDict[image_id] = []
                annotationsDict[image_id].append(annotation)

            IMAGEID = 0
            ANNOTATIONID = 0
            tList = []
            n_images = len(srcCOCODict["images"])
            aug_method = GlobalAugDict[getAugMethod(os.path.basename(eachWorkImageDir))]
            aug_number = getAugNum(os.path.basename(eachWorkImageDir))
            if n_images < N_THREADS:
                for i in range(n_images):
                    t = Thread(target=run, args=([srcCOCODict["images"][i]], ))
                    t.start()
                    tList.append(t)
            else:
                for i in range(N_THREADS):
                    mul = n_images//N_THREADS
                    if i == N_THREADS - 1:
                        t = Thread(target=run, args=(srcCOCODict["images"][i*mul:], ))
                    else:
                        t = Thread(target=run, args=(srcCOCODict["images"][i*mul:(i + 1)*mul], ))
                    t.start()
                    tList.append(t)
            for eachT in tList:
                eachT.join()

            with open(tarCOCOPath, "w",encoding="UTF-8") as fp:
                json.dump(tarCOCODict, fp,ensure_ascii=False)
                # return f"{os.path.basename(eachWorkImageDir)}目录完成数据增强!",img_aug
            if display == "是":
                return img_aug
            else:
                return []
            # print("{}目录完成数据增强!".format(os.path.basename(eachWorkImageDir)))

def image_aug(images,x1=-0.05,x2=0.05,y1=-0.05,y2=0.05,ml1=0.8,ml2=1.2,count=2, flip=0):
    #############################################################################
    # 注册 根据目录名使用的数据增强方法
    # imgaug官方文档
    # https://imgaug.readthedocs.io/en/latest/source/examples_bounding_boxes.html

    GlobalAugDict["global_default_aug_method"] = [
        iaa.Sequential([
            # iaa.OneOf([
            #     iaa.Affine(translate_px={"x":300}),
            #     iaa.Affine(translate_px={"x":-300}),
            # ])
            iaa.Fliplr(flip),
            iaa.Affine(
                translate_percent={"x":(x1, x2), "y":(y1, y2)},
            ),
            iaa.Multiply((ml1, ml2))
        ])
    ]

    #############################################################################
    # 注册 根据目录名, 每个目录每张图片的重复增强次数
    #############################################################################
    aug_method = GlobalAugDict["global_default_aug_method"]
    aug_number = count

    seq = iaa.Sequential(aug_method)

    # # 将Numpy数组转换为二进制数据
    # binary_data = images.tobytes()

    # # 使用cv2.imdecode对二进制数据进行解码
    # image = cv2.imdecode(np.frombuffer(binary_data, dtype=np.uint8), cv2.IMREAD_COLOR)

    aug_imgs = []
    try:
        for i in range(aug_number):
            image_aug= seq(image=images)
            aug_imgs.append(image_aug)
            # cv2.imencode(".jpg", aug_imgs)[1].tofile(os.path.join(aug_dir, "{file1}.jpg".format(i))
        return aug_imgs
    
    except Exception as e:
        print(e)

def check_dir(ROOTDIR):
    i = 0
    if os.path.exists(ROOTDIR):
        for eachImageDir in os.listdir(ROOTDIR):
            if os.path.isdir(os.path.join(ROOTDIR, eachImageDir)):
                if (cocoJsonName + ".json") in os.listdir(os.path.join(ROOTDIR, eachImageDir)):
                    augImageDir = os.path.join(ROOTDIR, eachImageDir + "_aug")
                    if os.path.exists(augImageDir):
                        if (cocoJsonName + ".json") in os.listdir(augImageDir):
                            return "数据安全警告: {}目录已存在{}, 数据增强过程中会覆盖该文件".format(eachImageDir + "_aug", cocoJsonName + ".json")
                    i += 1

        return "此次数据增强可应用到{}个目录".format(i)
    else:
        return "检查到路径异常,请校验!"

def test():
    with open('scripts/imgs_enhancement/1.jpg', 'rb') as file:
        binary_data = file.read()
        image = cv2.imdecode(np.frombuffer(binary_data, np.uint8), cv2.IMREAD_COLOR)

    a = image_aug(image,-0.05,0.05,-0.05,0.05,0.8,1.2,4)
    print(len(a))

# test()