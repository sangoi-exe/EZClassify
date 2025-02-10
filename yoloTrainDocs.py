from ultralytics import YOLO

YAML_PATH = r"E:/Dev/avmb/treinos/data.yaml"


def train_yolo_model():
    # Carrega o modelo YOLO pré-treinado
    model = YOLO("yolo11s.pt")

    # Inicia o treinamento usando os hiperparâmetros otimizados para mAP50 e mAP50-95.
    # Note que os resultados originais foram obtidos com 30 épocas, por isso ajustamos 'epochs=30'.
    model.train(
        data=YAML_PATH,
        epochs=100,  # Updated epochs
        imgsz=640,
        batch=16,
        workers=8,  # Updated workers
        project="Detect-Document",
        name="Model-v3",
        exist_ok=True,
        optimizer="SGD",  # Updated optimizer
        deterministic=False,
        rect=False,
        multi_scale=True,
        cos_lr=True,
        close_mosaic=0,
        profile=False,
        mask_ratio=4,
        dropout=0.1,
        val=False,  # Disabled validation
        plots=False,  # Disabled plots
        cache=False,
        resume=False,
        amp=True,
        pretrained=True,
        nms=True,
        # Optimizer hyperparameters updates:
        lr0=0.06339970827798422,  # Updated initial learning rate
        lrf=0.006418089082226196,  # Updated final learning rate factor
        momentum=0.02636258484162463,  # Updated momentum
        weight_decay=0.07508953718955161,  # Updated weight decay
        warmup_epochs=0.04624166591738838,  # Updated warmup epochs
        warmup_momentum=0.05034340056740538,  # Updated warmup momentum
        # Loss weights updates:
        box=0.06435276563832987,  # Updated box loss weight
        cls=0.07751982234739274,  # Updated class loss weight
        # Data augmentation parameters remain unchanged:
        hsv_h=0.03888,
        hsv_s=0.52486,
        hsv_v=0.70641,
        degrees=12.54,
        translate=0.86644,
        scale=0.29162,
        shear=5.33391,
        perspective=0.0003233,
        flipud=0.38589,
        fliplr=0.12862,
        bgr=0.79013,
        mosaic=0.22469,
        mixup=0.62582,
        copy_paste=0.14927,
        crop_fraction=0,
    )


if __name__ == "__main__":
    train_yolo_model()
