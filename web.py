import argparse

import gradio as gr
from transformers import pipeline, AutoConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuned model for text classification task")
    parser.add_argument("--cp", type=str, default="800", )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model_name = "./model_for_seqclassification/checkpoints/checkpoint-{}/".format(args.cp)

    title = "多标签文本分类"
    description = "输入文本，点击 submit 按钮"
    examples = [
        ["上海市人民政府关于完善困难残疾人生活补贴和重度残疾人护理补贴制度的实施意见", "基本养老服务/项目/特定情形老年保障"],
        ["2023年普陀区义务教育阶段学校规模和校舍场地条件表", "人口长期均衡发展/公共服务/义务教育, 招生入学/义务教育招生"],
        ["徐汇区关于支持元宇宙发展的若干意见", "惠企政策/产业发展, 惠企政策/创业支持, 营商环境/创新创业"],
    ]

    config = AutoConfig.from_pretrained(model_name)
    classifier = pipeline(task="text-classification", model=model_name, tokenizer=model_name)

    def do_classify(text):
        result = {}
        for x in classifier(text, top_k=len(config.id2label)):
            if len(result) < 5 or x["score"] > .3:
                result[x["label"]] = x["score"]
        return result

    demo = gr.Interface(
        do_classify,
        inputs=["text"],
        outputs=["label"],
        title=title,
        description=description,
        examples=examples,
    )
    demo.queue().launch(server_name="0.0.0.0", server_port=18080)

    # interface = gr.Interface.from_pipeline(
    #     classifier,
    #     title=title,
    #     description=description,
    #     examples=examples
    # )
    # interface.launch(server_name="0.0.0.0", server_port=18080)


if __name__ == "__main__":
    main()
