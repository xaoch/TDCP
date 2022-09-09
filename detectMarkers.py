from roboflow import Roboflow

rf = Roboflow(model_format="yolov5", notebook="roboflow-yolov5")

rf = Roboflow(api_key="PosvP3zTnv8pnd6MOQxG")
project = rf.workspace("tdcp").project("detect-lines")
dataset = project.version(5).download("yolov5",location="/scratch/xao1/TDCP/markers/")

# define number of classes based on YAML
import yaml
with open("/scratch/xao1/TDCP/markers/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])

print("Number of classes: {}".format(num_classes))

#python /home/xao1/Code/yolov5/train.py --img 416 --batch 16 --epochs 100 --data /scratch/xao1/TDCP/markers/data.yaml --cfg /scratch/xao1/TDCP/markers/models/custom_yolov5s.yaml --weights '' --name yolov5s_results  --cache