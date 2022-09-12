from roboflow import Roboflow
import os

rf = Roboflow(model_format="yolov5", notebook="ultralytics")
os.environ["DATASET_DIRECTORY"] = "/scratch/xao1/TDCP/datasets"

rf = Roboflow(api_key="PosvP3zTnv8pnd6MOQxG")
project = rf.workspace("tdcp").project("markers-lh8sr")
dataset = project.version(1).download("yolov5")

#!python train.py --img 416 --batch 16 --epochs 150 --data {dataset.location}/data.yaml --weights yolov5s.pt --cache
#python /home/xao1/Code/yolov5/train.py --img 416 --batch 16 --epochs 100 --data /scratch/xao1/TDCP/markers/data.yaml --cfg /scratch/xao1/TDCP/markers/models/custom_yolov5s.yaml --weights '' --name yolov5s_results  --cache

#python /home/xao1/Code/yolov5/detect.py --weights /home/xao1/Code/yolov5/runs/train/yolov5s_results/weights/best.pt --source /scratch/xao1/TDCP/smaller.mp4 --project /scratch/xao1/TDCP/markers/results/