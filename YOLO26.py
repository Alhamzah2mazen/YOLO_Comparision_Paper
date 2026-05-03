from ultralytics import YOLO

def train_v26():
    # Load your custom YOLO26 weights
    model = YOLO('yolo26n.pt') 

    model.train(
        data='Data/Behaviour/data.yaml',
        epochs=50,
        imgsz=640,
        device=0,
        workers=2,
        batch=16,
        
        # --- Pure Architecture Settings ---
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        # ----------------------------------
        
        project='UJ_Graduation_Project',
        name='v26_Raw_Evolution'
    )

if __name__ == '__main__':
    train_v26()