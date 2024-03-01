import gradio as gr
from coco_label import print_statistics


def filter_records(records, gender):
    return records[records["gender"] == gender]

res = print_statistics(r"Y:\label\label_P-IJC23110092_岚图总装检测\车门工位\标注图\MJSB标签取出")
demo = gr.Interface(
    filter_records,
    [
        gr.Dataframe(
            value = res,
            headers=["算法标签", "标签数量", "图片数量","平均占比"],
            datatype=["str", "str", "str","str"],
            row_count=10,
            col_count=(4, "fixed"),
        ),
    ],
    "dataframe",
    description="",
)

if __name__ == "__main__":

    demo.launch()