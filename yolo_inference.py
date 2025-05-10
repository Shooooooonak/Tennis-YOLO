from ultralytics import YOLO

model = YOLO('yolov8x.pt')  # Load a pretrained YOLOv8 nano model

trained_model = YOLO('models/yolov5nu_best.pt')  # Load a custom trained YOLOv5 model

result = trained_model.predict('input_videos/input_video.mp4', conf = 0.2, save =True)  # Predict on an image and save the result
print(result)
print("boxes:")
for box in result[0].boxes:
    print(box)