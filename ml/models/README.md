# ML Models Directory

This directory contains pre-trained model weights for license plate detection and OCR.

## Required Models

### 1. YOLOv8 Model
The base YOLOv8 model is automatically downloaded when the application starts.

For better license plate detection, you can use a fine-tuned model:
- Place your custom weights as `license_plate_detector.pt`

### 2. EasyOCR Models
OCR models are automatically downloaded and cached by EasyOCR.
Default languages: English (`en`) and Arabic (`ar`)

## Download Models

Run the download script:
```bash
python scripts/download_models.py
```

## Custom License Plate Model

For improved accuracy, consider training a custom YOLO model on license plate datasets:

1. **Datasets:**
   - [CCPD (Chinese City Parking Dataset)](https://github.com/detectRecog/CCPD)
   - [OpenALPR Benchmarks](https://github.com/openalpr/benchmarks)
   - [UFPR-ALPR Dataset](https://web.inf.ufpr.br/vri/databases/ufpr-alpr/)

2. **Training:**
   ```python
   from ultralytics import YOLO

   model = YOLO('yolov8n.pt')
   model.train(data='license_plate.yaml', epochs=100)
   ```

3. **Place the trained model:**
   ```
   ml/models/license_plate_detector.pt
   ```

## File Structure

```
ml/models/
├── README.md               # This file
├── .gitkeep               # Keep directory in git
├── license_plate_detector.pt  # (Optional) Custom YOLO weights
└── (other model files)
```
