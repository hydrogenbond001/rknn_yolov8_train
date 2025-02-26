from ultralytics import YOLO

def train_yolov8():
    # 加载预训练模型
    model = YOLO('yolov8n.pt')  # 可以选择 yolov8n, yolov8s, yolov8m, yolov8l, yolov8x

    # 开始训练
    results = model.train(
        data='/home/ph/ultralytics_yolov8/train/gonxun.v7i.yolov8/data.yaml',# 数据集配置文件路径
        epochs=10,        # 训练轮数
        imgsz=640,         # 输入图像大小
        batch=16,          # 批量大小
        lr0=0.01,          # 初始学习率
        augment=True,      # 是否启用数据增强
        name='yolov8'  # 训练任务名称
    )

    # 返回训练好的模型
    return model

def validate_model(model):
    # 验证模型
    metrics = model.val()  # 使用验证集评估模型
    print(f"mAP50-95: {metrics.box.map}")  # 打印 mAP 指标
    print(f"mAP50: {metrics.box.map50}")

def predict_with_model(model, image_path):
    # 使用模型进行推理
    results = model.predict(source=image_path, save=True, conf=0.5)  # conf 为置信度阈值
    for result in results:
        print(result.boxes)  # 打印检测到的目标框信息

def export_model(model):
    # 导出模型为 ONNX 格式
    model.export(format='onnx')
    print("模型已导出为 ONNX 格式")

if __name__ == "__main__":
    # 训练模型
    trained_model = train_yolov8()

    # 验证模型
    #validate_model(trained_model)

    # 使用模型进行推理
    #predict_with_model(trained_model, 'path/to/image.jpg')  # 替换为你的测试图像路径

    # 导出模型
    #export_model(trained_model)
# 导出模型需要把default.yaml里的model: yolov8n.pt改为你的模型地址