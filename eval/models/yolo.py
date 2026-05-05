from networkx import config
from ultralytics import YOLO
import torch
from typing import Union
from eval.models.base import BaseModel

class YOLOHumanDetector(BaseModel):
    """
    YOLOHumanDetector
    -----------------
    Unified interface for human-presence detection using YOLOv8.
    Returns "1" if a person is detected above the confidence threshold, else "0".

    Parameters
    ----------
    model_name : str, optional
        YOLO model variant to load (default: 'yolov8s.pt').
    device : str or torch.device, optional
        Inference device ('cuda' or 'cpu'). Auto-detected if None.
    conf_thresh : float, optional
        Minimum confidence threshold to count a detection as valid.
    iou_thresh : float, optional
        IOU threshold for non-max suppression.
    """

    def __init__(self,config):
        super().__init__(config)
        
        self.conf_thresh = 0.6 if config.conf_thresh is None else config.conf_thresh
        self.iou_thresh = 0.45 if config.iou_thresh is None else config.iou_thresh
        self.model_name = config.model_name
        self.conf_thresh = config.conf_thresh
        self.iou_thresh = config.iou_thresh
        self.device_map = config.device_map

        # Load YOLO model
        self.model = YOLO(self.model_path)
        self.model.to(self.device)

    def predict(self, image_path: str, prompt: str = "") -> str:
        """
        Run YOLO inference and return "1" if a human is detected, otherwise "0".

        Parameters
        ----------
        image_path : str
            Path to the input image.
        prompt : str, optional
            Placeholder for interface compatibility (ignored).

        Returns
        -------
        str
            "1" if human detected above threshold, otherwise "0".
        """
        results = self.model.predict(
            source=image_path,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            verbose=False
        )[0]

        for box in results.boxes:
            cls = self.model.names[int(box.cls)]
            conf = float(box.conf)
            if cls.lower() == "person" and conf >= self.conf_thresh:
                return "1"

        return "0"
