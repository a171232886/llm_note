import transformers

from datasets import Dataset, load_dataset, load_from_disk
from transformers import OPTForCausalLM, GPT2Tokenizer
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model

from config import trainer_args, parser_arg
from setlog import setlog


def process_dataset(tokenizer, data_load_from_disk=True, debug=False):
    if data_load_from_disk == False:
        """
        载入数据
        """
        raw_dataset = load_dataset("Dahoas/rm-static")
        raw_train_dataset = raw_dataset["train"]
        raw_eval_dataset = raw_dataset["test"]

        # print(raw_train_dataset[0]) 查看每条数据
        # print(raw_train_dataset.features) 查看每条数据构成类型

        """
        为将每条数据转换成编码，使用tokenizer
        每句话经过tokenizer转换后，输出字典{'input_ids', 'attention_mask'}
            - input_ids 为每个单词在词库中的序号
            - attention_mask 1表示该位置为单词，0表示为pad
        """
        
        # # 查看tokenizer对一句话的处理
        # inputs = tokenizer(raw_train_dataset[0]['prompt']) 
        # print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))

        
        # 改变报错级别
        transformers.logging.set_verbosity_error()

        # 此种方式相当于for循环处理，将'input_ids', 'attention_mask'添加到每条数据中
        def tokenize_function(example):
            # 注意padding，使每条数据长度保持一致
            tokenized_data = tokenizer(example["prompt"], example["chosen"], 
                            max_length=512,
                            padding="max_length",
                            truncation=True)
            tokenized_data["label"] = tokenized_data["input_ids"] 
            return tokenized_data
        # batched=True是为了加速
        tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

        # 保存处理好的数据
        tokenized_dataset.save_to_disk("./data")
    else:
        tokenized_dataset = load_from_disk("./data")

    # 检查每条数据的长度是否一致
    # tokenized_train_dataset = tokenized_dataset["train"][:20]
    # tokenized_train_dataset = {k:v for k,v in tokenized_train_dataset.items() if k in ['attention_mask', 'input_ids'] }
    # print([len(v) for v in tokenized_train_dataset['input_ids']])

    if debug == False:
        tokenized_train_dataset = tokenized_dataset["train"]
        tokenized_eval_dataset = tokenized_dataset["test"]
    else:
        # 裁剪数据长度，dubug时使用
        tokenized_train_dataset = Dataset.from_dict(tokenized_dataset["train"][:100])
        tokenized_train_dataset = Dataset.from_dict(tokenized_dataset["test"][:100])

    return tokenized_train_dataset, tokenized_train_dataset

def main():
    args = parser_arg()
    # set log
    logger = setlog()
    logger.warning("Start")

    # tokenizer, model, dataset
    tokenizer = GPT2Tokenizer.from_pretrained("/home/dell/wh/model/opt-1.3b")
    model = OPTForCausalLM.from_pretrained("/home/dell/wh/model/opt-1.3b")
    tokenized_train_dataset, tokenized_train_dataset = process_dataset(tokenizer, 
                                                                       data_load_from_disk=args.data_load_from_disk, 
                                                                       debug=args.debug)

    # LLama x LoRA
    lora_config = LoraConfig(
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Trainer
    training_args = TrainingArguments(**trainer_args)

    trainer = Trainer(model=model, 
                      tokenizer=tokenizer, 
                      args=training_args, 
                      train_dataset=tokenized_train_dataset,
                      eval_dataset=tokenized_train_dataset)
    result = trainer.train()

if __name__ == "__main__":
    main()