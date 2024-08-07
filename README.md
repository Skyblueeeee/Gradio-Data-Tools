![](ui/imgs/mt_logo.jpg)

本平台主要基于Gradio Python库构建，旨在为用户提供一个通用、高效的脚本执行环境。作为模型训练的第一站，我们支持多人多任务同时操作，确保工作效率最大化。

<center><b>Gradio</b> <a href="https://www.gradio.app/"> ✨✨ </a> ｜  <b>FiFtyone</b> <a href="https://github.com/voxel51/fiftyone"> ✨✨ </a>&nbsp;｜ <b>MMDET</b> <a href="https://github.com/open-mmlab/mmdetection"> ✨✨</a>&nbsp;｜ <b>Github</b> <a href="https://github.com/Skyblueeeee/Gradio-Data-Tools"> ✨✨</a>&nbsp; </center>
## 核心功能特色:

**✦Fiftyone数据管理**：灵活导入数据，实现数据库的精细化管理。支持数据的导出、删除，以及有效与空图的快速检查。标签校验与发标注图等任务一键完成。

✦**AI自动标注**：支持手动上传模型进行预标注，自动生成via与coco文件，可一键执行数据分析与数据管理任务。

**✦数据统计分析**：对COCO格式文件进行深入分析，精确统计图片与标签的数量及其分布，为模型训练前提供有力支持。

**✦离线数据增强**：提供丰富多样的数据增强选项，包括偏移、明亮度调整、翻转等，支持从单张到多张图片的灵活增强策略。

**✦提取标注标签**：可选择保留图像中现有标签，或进行转换，支持一键导出为COCO格式，满足不同需求。

**✦JSON文件处理**：提供VIA与COCO格式之间转换，以及VIA和YOLO格式之间的转换。via2coco转换特别优化，自动过滤无效和空图，确保image_id为整数型。

平台将持续迭代，引入更多实用功能，助力您的数据工作流程更加流畅和高效。

## 更新日志

### 待定功能
【数据管理】[入库模式]增加TXT格式。

### 20240709-修复BUG

【数据管理】[数据库管理]增加关键词删除，可正常使用不同的删除模式。

【数据分析】修复统计的图片利用数量错误问题。

### 20240426-更新功能

【JSON互转】增加mask的json互转。

### 20240312-更新功能

【AI预标注】模块上线，支持手动上传模型进行推理标注，自动标注完成后生成via与coco文件。[测试中]

### 20240305-更新功能

【数据管理】增加[数据库录入]原图默认tag、强制图片名去重；增加目录条件检索、[数据库发图]模板填写功能。

【数据分析】增加目录条件检索。

【JSON互转】增加YOLO转VIA格式。

### 20240227-修复BUG

【提取标签】已解决选择不保留其他标签时，导致保留的多个标签只有一个。

【提取标签】增加可输出图片长宽选项，当转coco文件时，输入的路径via文件并没有宽高字段，且选择未复制图片，需要手动输入长宽。

### 20240226-更新功能

【提取标签】增加是否复制图片选项。

### 20240125-初版发布
【全平台】精简代码，细分各个模块功能，优化界面布局。

### 20231228-设计测试
设计-编写-调试。

完成之前小工具的脚本迁移，通用性测试。

## 快速开始

1.创建虚拟环境：

```
conda create -n py38_gdt python==3.8
```

2.安装torch、torchvision

```
下载Torch、Torchvision：torch-1.11.0%2Bcu115-cp38   torchvision-0.12.0%2Bcu115-cp38
https://download.pytorch.org/whl/torch_stable.html

cd Gradio-Data-Tools

pip install 下载的torch

pip install 下载的torchvision

pip install -r requirements.txt
```

3.更改快捷启动虚拟环境路径 :

```
PYTHON_PATH=..\python_env\py38_gdt\python.exe
```

4.双击启动!
