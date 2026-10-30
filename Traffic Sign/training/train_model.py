"""Train actual four-class YOLOv10n weights on CPU; outputs stay outside checkout."""
import os
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("MKL_NUM_THREADS","4")
os.environ.setdefault("YOLO_CONFIG_DIR","/workspace/traffic-sign-training/config")
import argparse
import json
from pathlib import Path
import torch
from ultralytics import YOLO

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",default="/workspace/traffic-sign-training/dataset/dataset.yaml")
    parser.add_argument("--weights",default=str(Path(__file__).resolve().parents[2]/"benchmarking/models/yolov10n.pt"))
    parser.add_argument("--epochs",type=int,default=25)
    parser.add_argument("--name",default="four_signs")
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model=YOLO(args.weights)
    model.train(data=args.data,epochs=args.epochs,imgsz=320,batch=8,device="cpu",workers=0,
                project="/workspace/traffic-sign-training/runs",name=args.name,exist_ok=False,
                freeze=10,optimizer="AdamW",lr0=.001,weight_decay=.0005,patience=10,
                seed=20261030,deterministic=True,amp=False,cache=False,plots=True,
                fliplr=0.,flipud=0.,degrees=0.,perspective=0.,mosaic=.5,close_mosaic=5,
                mixup=0.,copy_paste=0.,hsv_h=.01,hsv_s=.3,hsv_v=.25)
    best=Path(model.trainer.best)
    trained=YOLO(str(best))
    test=trained.val(data=args.data,split="test",imgsz=320,batch=8,device="cpu",workers=0,
                     project="/workspace/traffic-sign-training/evaluation",name=args.name,plots=True)
    summary={"best_weights":str(best),"test_metrics":test.results_dict,
             "scope":"Synthetic heldout test, NOT physical camera/road accuracy or Pi benchmark."}
    (best.parent.parent/"heldout_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
