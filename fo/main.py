from logging import config
from trainer import BuddhistPOSTrainer
from predictor import BuddhistPOSPredictor

base_config = {
    # paths
    "model_path": "model/sikubert",
    "custom_dict_path": "data/fo_dict.txt",
    "model_save_path": "model/buddhist_pos_model",

    # training hyperparameters
    "num_train_epochs": 5,
    "train_batch_size": 8,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "adam_epsilon": 1e-8,
    "warmup_steps": 0,
    "logging_steps": 100,
    "save_steps": 1000,
    "evaluation_strategy": "steps",
    "eval_steps": 1000,
    "save_total_limit": 1,
    "no_cuda": False,
    "seed": 42,

    # model hyperparameters
    "max_seq_length": 128,
    "num_labels": 17,
    "use_crf": True,
    "use_biaffine": True,
    "dropout": 0.1
}


def main():
    # 示例：训练模型
    sentences = []
    with open("data/siku本地断句/C1710_001_result.txt", "r", encoding="utf-8") as f:
        for line in f:
            sentences.append(line.strip())

    config = base_config.copy()
    # 初始化训练器
    trainer = BuddhistPOSTrainer(
        model_name=config["model_path"],
        custom_dict_path=config["custom_dict_path"]  # 您的佛教词典
    )

    # # 开始训练
    # trainer.train(
    #     sentences,
    #     output_dir=config["model_save_path"],
    #     num_train_epochs=config["num_train_epochs"],
    #     train_batch_size=config["train_batch_size"],
    #     # learning_rate=config["learning_rate"],
    #     # weight_decay=config["weight_decay"],
    #     # adam_epsilon=config["adam_epsilon"],
    #     # warmup_steps=config["warmup_steps"],
    #     # logging_steps=config["logging_steps"],
    #     # save_steps=config["save_steps"],
    #     # evaluation_strategy=config["evaluation_strategy"],
    #     # eval_steps=config["eval_steps"],
    #     # save_total_limit=config["save_total_limit"],
    #     # no_cuda=config["no_cuda"],
    #     # seed=config["seed"],
    #     # max_seq_length=config["max_seq_length"],
    #     # num_labels=config["num_labels"],
    #     # use_crf=config["use_crf"],
    #     # use_biaffine=config["use_biaffine"],
    #     # dropout=config["dropout"]
    # )

    # # 使用训练好的模型进行预测
    # predictor = BuddhistPOSPredictor(config["model_save_path"])

    # test_text = "海水住於一味即攝眾味住於大海即混諸流如人在大海中浴即用一切水"
    # results = predictor.predict(test_text)

    # print("词性标注结果:")
    # for word, pos in results:
    #     print(f"{word}/{pos}")


if __name__ == "__main__":

    # 运行训练和预测
    main()