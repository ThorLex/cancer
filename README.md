# Breast Cancer Detection System

An AI-powered platform for detecting suspicious masses in mammography images, designed for clinical integration and research applications.

## About This Project

This system started as a research initiative to train deep learning models on publicly available mammography datasets. After achieving promising results, we expanded it into a complete platform with API services and tools that can integrate with existing clinical workflows. The system has been successfully deployed in local hospitals using their own labeled data.

The platform handles the entire pipeline from raw medical images to actionable predictions, with built-in tools for visualization, evaluation, and explanation of model decisions.

## Requirements

You need to install CUDA and PyTorch manually before proceeding with the rest of the installation. This prevents dependency conflicts that occur with automatic installation.

**Install CUDA drivers for your system:**

Linux users can install via package manager:
```bash
apt install nvidia-cuda-toolkit
```

Windows and Linux users can also download directly from NVIDIA:
https://developer.nvidia.com/cuda-downloads

**Install PyTorch with CUDA support:**

Visit the PyTorch website and follow the installation guide for your specific setup:
https://pytorch.org/get-started/locally/

**Install system dependencies:**

Linux:
```bash
sudo apt install libmagic1
```

Windows:
```bash
pip install python-magic-bin
```

If using Docker, install the container toolkit on your host machine:
```bash
sudo apt install nvidia-container-toolkit
```

## Installation Steps

Download the project:
```bash
git clone https://github.com/monajemi-arman/breast_cancer_detection
cd breast_cancer_detection
```

Install Python packages:
```bash
pip install --no-build-isolation -r requirements.txt
```

## Preparing Your Data

### Dataset Information

The system works with three established mammography datasets:

- InBreast
- CBIS-DDSM
- MIAS

### Getting the Datasets

**Automatic download using Google Colab:**

Open the download_datasets_colab.ipynb notebook in Google Colab. The notebook will prompt you to upload your kaggle.json credentials file. You can download this file from your Kaggle account settings at https://www.kaggle.com/settings under the API section.

**Manual download:**

Download from these sources:
- https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset
- https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset
- https://www.kaggle.com/datasets/kmader/mias-mammography

After downloading, create this folder structure:
```
breast_cancer_detection/
  datasets/
    all-mias/
    CBIS-DDSM/
      csv/
      jpeg/
    INbreast Release 1.0/
      AllDICOMs/
```

Place each downloaded dataset in its corresponding folder.

### Converting to Standard Format

Run the conversion script to process all datasets into a unified format:
```bash
python convert_dataset.py
```

This creates standardized folders and files that the training scripts can use:
- images/
- labels/
- dataset.yaml
- annotations.json

### Checking Your Data

View your processed images with annotations overlaid:

```bash
python visualizer.py -m coco -d train/images -l train.json
```

Or if you're using YOLO format:
```bash
python visualizer.py -m yolo -d train/images -l train/labels
```

### Enhancing Images

Apply preprocessing filters if needed. Available options are canny, clahe, gamma, histogram, and unsharp:

```bash
python filters.py -i INPUT_FOLDER -o OUTPUT_FOLDER -f FILTER_NAME
```

## Training Your Model

### Faster R-CNN Approach

Start training:
```bash
python detectron.py -c train
```

Test your trained model on an image:
```bash
python detectron.py -c predict -w output/model_final.pth -i YOUR_IMAGE.jpg
```

Check model performance metrics:
```bash
python detectron.py -c evaluate -w output/model_final.pth
```

Export predictions for detailed analysis:
```bash
python detectron.py -c evaluate_test_to_coco -w output/model_final.pth
```

### YOLO Approach

Install YOLO framework:
```bash
pip install ultralytics
```

Train a model:
```bash
yolo train data=dataset.yaml model=yolov8n
```

Run predictions:
```bash
yolo predict model=runs/detect/train/weights/best.pt source=images/cb_1.jpg conf=0.1
```

## Running the Web Interface

### Setting Up

After training completes, prepare the web application:

1. Look in the output/last_checkpoint file to find your final model filename
2. Copy detectron.cfg.pkl to the webapp folder
3. Copy your model checkpoint to webapp/model.pth

### Starting the Server

```bash
cd webapp/
python web.py
```

Open your browser and go to:
http://127.0.0.1:33517

### Using the API

Send an image for analysis:
```bash
curl -X POST \
  -F "file=@input.jpg" \
  http://localhost:33517/api/v1/predict \
  | jq -r '.data.inferred_image' | base64 --decode > prediction.jpg
```

Process multiple images at once:
```bash
curl -X POST \
  -F "file=@image1.jpg" \
  -F "file=@image2.jpg" \
  http://localhost:33517/api/v1/predict
```

## Clinical Integration Services

### DICOM File Handling

Convert medical DICOM files to JPEG format:
```bash
curl -X POST -F 'file=@YOUR_DICOM_FILE' http://localhost:33521/upload
```

The service automatically detects whether files are compressed or uncompressed.

### Hash-Based File Routing

This service eliminates redundant file uploads in multi-step workflows. After converting a DICOM file, use its hash to reference it in subsequent API calls:

```bash
curl -X POST "http://localhost:33516/route" \
     -H "Content-Type: application/json" \
     -d '{
          "hash": "ee4daa5e0a8065c4d51be25ef233cdd276bca34de5a36ebc3406c8a82dd41c2a",
          "data": { 
              "file": "ee4daa5e0a8065c4d51be25ef233cdd276bca34de5a36ebc3406c8a82dd41c2a"
          },
          "endpoint": "http://localhost:33517/api/v1/predict"
     }'
```

### Automated Folder Monitoring

Monitor a directory for new DICOM files and process them automatically:

```bash
curl "http://localhost:33522/images"
curl "http://localhost:33522/images?count=10&page=1"
curl "http://localhost:33522/hash_to_original?hash=HASH_VALUE"
```

## Understanding Model Decisions

The explainable AI component generates visual heatmaps showing which areas of an image influenced the model's decision.

### Preparation

Convert your detection dataset to classification format:
```bash
python coco_to_classification.py train.json train_class.json
```

Train the classification model:
```bash
python classification_model.py -a train_class.json -d train/images --save_dir classification_output -c train
```

### Generating Explanations

Create a visualization for a specific image:
```bash
python classification_model.py --save_dir classification_output -c predict -i train/images/YOUR_IMAGE.jpg
```

### Running as a Service

Start the explanation API:
```bash
python classification_model.py --save_dir classification_output -c api
```

Get an explanation heatmap:
```bash
curl -X POST -F "file=@test/images/20586986.jpg" http://localhost:33519/predict | jq -r '.activation_map' | base64 -d > heatmap.jpg
```

## AI Assistant Integration

The chatbot service uses language models to discuss predictions and answer questions about mammography results.

### Configuration

Copy the template and add your API credentials:
```bash
cp llm/config.json.default llm/config.json
```

Edit llm/config.json with your OpenAI API key.

### Running the Service

```bash
python llm/llm_api_server.py
```

### Interacting with the Assistant

```bash
curl -X POST http://localhost:33518/generate-response \
-H "Content-Type: application/json" \
-d '{
  "prompt": "What is BI-RADS 4?",
  "predictions": "PREDICTION_DATA_HERE"
}'
```

## Project Organization

**Core Training Scripts:**
- detectron.py - Faster R-CNN training and inference
- classification_model.py - Classification and explainability

**Data Processing:**
- convert_dataset.py - Dataset standardization
- coco_to_classification.py - Format conversion
- filters.py - Image preprocessing
- visualizer.py - Data inspection

**Web Services:**
- webapp/ - Main prediction interface
- llm/ - Chatbot integration

**Clinical Tools:**
- DICOM conversion service
- Hash routing service
- Folder monitoring service

## Supported Frameworks

The platform supports multiple deep learning approaches:

- Faster R-CNN via Detectron2
- YOLO family of models
- Any framework that accepts COCO or YOLO format datasets
- Custom implementations like UaNet for specialized use cases

## Technical Notes

The system includes multiple API services that can run concurrently. Each service operates on a different port and handles a specific aspect of the clinical workflow. This modular design allows you to use only the components you need for your particular deployment.

All services are designed to handle production workloads and include appropriate error handling for clinical environments where reliability is critical.
