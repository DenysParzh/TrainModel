import comet_ml
import torch
from ultralytics import YOLO


def train():
    API_KEY = ""
    PROJECT_NAME = "drone-detector"
    MODEL_PATH = "yolo11s.pt"
    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    model = YOLO(MODEL_PATH)

    torch.cuda.empty_cache()
    comet_ml.init(api_key=API_KEY, project_name=PROJECT_NAME)

    # model.train(data='config.yaml',
    #             project=PROJECT_NAME,
    #             epochs=1,
    #             batch=30,
    #             imgsz=640,
    #             device=device,
    #             save_period=10,
    #             workers=12,
    #             box=7.5,
    #             hsv_h=0.015,
    #             hsv_s=0.7,
    #             hsv_v=0.4,
    #             translate=0.3,
    #             scale=0.5,
    #             shear=0.05,
    #             degrees=0.1,
    #             erasing=0.2,
    #             flipud=0.5,
    #             fliplr=0.5
    #             )

    model.tune(data='config.yaml', 
                epochs=30,
                iterations=300, 
                optimizer="AdamW", 
                plots=False, 
                save=False,
                val=False)
    

if __name__ == "__main__":
    train()
