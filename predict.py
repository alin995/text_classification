import argparse

from transformers import pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuned model for text classification task")
    parser.add_argument("--model_name_or_path", type=str, default="./model_for_seqclassification/checkpoints/checkpoint-200/",
                        help="Path to pretrained model or model identifier from huggingface.co/models.", )
    args = parser.parse_args()
    return args


def main():
    model_name_or_path = "./model_for_seqclassification/checkpoints/checkpoint-200/"
    classifier = pipeline(task="text-classification", model=model_name_or_path, tokenizer=model_name_or_path)
    result = classifier("苹果", top_k=4)
    print(result)
    print(classifier("橙子", top_k=4))
    print(classifier("榴莲", top_k=4))
    print(classifier("螺蛳粉", top_k=4))
    print(classifier("酸辣粉", top_k=4))
    print(classifier("火锅", top_k=4))
    print(classifier("酸辣肥牛", top_k=4))
    print(classifier("榴莲、螺蛳粉", top_k=4))

    # a = TextClassificationPipeline()


if __name__ == "__main__":
    main()
