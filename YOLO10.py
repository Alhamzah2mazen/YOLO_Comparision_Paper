from ultralytics import YOLO

def train_v10():
    # Load the official YOLOv10 small or nano model
    model = YOLO('yolov10s.pt') 

    model.train(
        data='Data/Behaviour/data.yaml',
        epochs=50,
        imgsz=640,
        device=0,
        workers=2,         # Kept low to avoid WinError 1455
        batch=16,          # Optimized for your 16GB VRAM
        
        # --- Pure Architecture Settings ---
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        # ----------------------------------
        
        project='UJ_Graduation_Project',
        name='v10_Raw_Evolution'
    )

if __name__ == '__main__':
    train_v10()