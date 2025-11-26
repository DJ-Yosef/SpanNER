from .trainer import BuddhistPOSTrainer
from .predictor import BuddhistPOSPredictor

def evaluate_model(model_path, test_sentences, true_labels):
    """
    评估模型性能
    """
    predictor = BuddhistPOSPredictor(model_path)
    predictions = predictor.batch_predict(test_sentences)

    # 计算准确率等指标
    from seqeval.metrics import classification_report

    # 将预测结果和真实标签转换为适合seqeval的格式
    # 这里需要根据您的数据格式进行调整

    print(classification_report(true_labels, predictions))

def fine_tune_with_more_data(base_model_path, new_sentences, output_dir):
    """
    使用更多数据继续微调模型
    """
    trainer = BuddhistPOSTrainer()
    trainer.load_model(base_model_path)

    # 继续训练
    trainer.train(
        new_sentences,
        output_dir=output_dir,
        num_train_epochs=3,  # 较少的epochs进行微调
        learning_rate=1e-5,  # 较小的学习率
    )