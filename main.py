from detector import *
import os

def main():
    #videoPath = 0
    videoPath = "http://192.168.123.12:8080/?action=stream"
    configPath = os.path.join("model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data", "frozen_inference_graph.pb")
    classesPath = os.path.join("model_data", "coco.names")

    det = Detector(videoPath, configPath, modelPath, classesPath)
    det.onVideo()

if __name__ == '__main__':
    main()