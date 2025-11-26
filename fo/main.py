from logging import config
from .trainer import BuddhistPOSTrainer
from .predictor import BuddhistPOSPredictor

base_config = {
    # paths
    "model_path": "model/sikubert",
    "custom_dict_path": "buddhist_terms.txt"
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
    "dropout": 0.1,
}


def main():
    # 示例：训练模型
    sentences = [
        "海水住於一味即攝眾味住於大海即混諸流",
        "如人在大海中浴即用一切水所以聲聞悟迷凡夫迷悟",
        "聲聞不知聖心本無地位因果階級心量妄想修囙證果",
        "佛告文殊師利菩薩一切眾生皆有佛性",
        "般若波羅蜜多心經觀自在菩薩行深般若波羅蜜多時",
        # 添加更多训练句子...
    ]

    config = base_config.copy()
    # 初始化训练器
    trainer = BuddhistPOSTrainer(
        model_name=config["model_path"],
        custom_dict_path=config["custom_dict_path"]  # 您的佛教词典
    )

    # 开始训练
    trainer.train(
        sentences,
        output_dir=config["model_save_path"],
        num_train_epochs=config["num_train_epochs"],
        train_batch_size=config["train_batch_size"],
        # learning_rate=config["learning_rate"],
        # weight_decay=config["weight_decay"],
        # adam_epsilon=config["adam_epsilon"],
        # warmup_steps=config["warmup_steps"],
        # logging_steps=config["logging_steps"],
        # save_steps=config["save_steps"],
        # evaluation_strategy=config["evaluation_strategy"],
        # eval_steps=config["eval_steps"],
        # save_total_limit=config["save_total_limit"],
        # no_cuda=config["no_cuda"],
        # seed=config["seed"],
        # max_seq_length=config["max_seq_length"],
        # num_labels=config["num_labels"],
        # use_crf=config["use_crf"],
        # use_biaffine=config["use_biaffine"],
        # dropout=config["dropout"]
    )

    # 使用训练好的模型进行预测
    predictor = BuddhistPOSPredictor(config["model_save_path"])

    test_text = "海水住於一味即攝眾味住於大海即混諸流如人在大海中浴即用一切水"
    results = predictor.predict(test_text)

    print("词性标注结果:")
    for word, pos in results:
        print(f"{word}/{pos}")

def create_buddhist_dict():
    """
    创建佛教专用词典的示例
    """
    buddhist_terms = """
    般若 n
    波罗蜜 n
    菩提 n
    涅槃 n
    佛性 n
    如来藏 n
    阿赖耶识 n
    真如 n
    法界 n
    因果 n
    业力 n
    轮回 n
    净土 n
    禅定 n
    文殊师利 nh
    观自在 nh
    阿弥陀佛 nh
    释迦牟尼 nh
    金刚经 ns
    心经 ns
    法华经 ns
    摩诃般若波罗蜜 n
    阿耨多罗三藐三菩提 n
    """

    with open("buddhist_terms.txt", "w", encoding="utf-8") as f:
        f.write(buddhist_terms)

    print("已创建佛教词典: buddhist_terms.txt")

if __name__ == "__main__":
    # 创建示例词典
    create_buddhist_dict()

    # 运行训练和预测
    main()