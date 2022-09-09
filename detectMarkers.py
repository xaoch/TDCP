from roboflow import Roboflow

rf = Roboflow(model_format="yolov5", notebook="roboflow-yolov5")

rf = Roboflow(api_key="PosvP3zTnv8pnd6MOQxG")
project = rf.workspace("tdcp").project("detect-lines")
dataset = project.version(5).download("yolov5",location="/scratch/xao1/TDCP/markers/")

