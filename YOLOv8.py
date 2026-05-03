from ultralytics import YOLO

def train_model():
    # 1. Load the model
    model = YOLO('yolov8n.pt') 

    # 2. Start Training on your RTX 5060 Ti
    results = model.train(
        data='Data/Behaviour/data.yaml', # Path from your terminal output
        epochs=50,
        imgsz=640,
        device=0,
        workers=0,                       # This is what caused the error
        
        # --- Pure Architecture Settings ---
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        # ----------------------------------
        
        project='UJ_Graduation_Project',
        name='v8_Raw_Evolution'
    )

# This "if" block is the required idiom for Windows multiprocessing
if __name__ == '__main__':
    train_model()