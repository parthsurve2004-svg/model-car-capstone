"""Evaluate the learned detector on heldout synthetic data and printable cards.

Results describe synthetic-domain behavior, never physical camera reliability.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("YOLO_CONFIG_DIR", "/workspace/traffic-sign-training/config")
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from car.perception.traffic_signs import suppress_duplicate_signs


def predictions(model, image, confidence):
    result=model.predict(image,imgsz=320,conf=confidence,device="cpu",verbose=False)[0]
    return result,[{"class_name":result.names[int(b.cls[0])],"class_id":int(b.cls[0]),
                    "confidence":float(b.conf[0]),"bbox":b.xyxy[0].tolist(),
                    "type":"traffic_sign"} for b in result.boxes]


def annotate(image, items):
    canvas=image.copy()
    for item in items:
        x1,y1,x2,y2=map(int,item["bbox"])
        cv2.rectangle(canvas,(x1,y1),(x2,y2),(255,160,20),2)
        cv2.putText(canvas,f"{item['class_name']} {item['confidence']:.2f}",(x1,max(15,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX,.6,(255,100,20),2)
    return canvas


def card_checks(model, dataset, output, confidence):
    output.mkdir(parents=True,exist_ok=True)
    summary={}
    for path in sorted((dataset/"printable_cards").glob("*.png")):
        native=cv2.imread(str(path))
        padded=np.full((480,640,3),255,np.uint8)
        padded[160:320,240:400]=cv2.resize(native,(160,160))
        for label,image in ((path.stem+"_native",native),(path.stem+"_padded",padded)):
            result,items=predictions(model,image,confidence)
            summary[label]=items
            cv2.imwrite(str(output/(label+"_raw.jpg")),result.plot())
            cv2.imwrite(str(output/(label+".jpg")),annotate(image,suppress_duplicate_signs(items)))
    generator=runpy.run_path(str(Path(__file__).with_name("generate_dataset.py")))
    fonts=[Path("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"),
           Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
           Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
           Path("/usr/share/fonts/truetype/open-sans/OpenSans-Bold.ttf"),
           Path("/usr/share/fonts/truetype/open-sans/OpenSans-CondBold.ttf")]
    for font in fonts:
        if not font.exists():
            continue
        generator["card"].__globals__["FONT_PATH"]=str(font)
        card=generator["card"](0,random.Random(97013),printable=True)
        native=cv2.cvtColor(card,cv2.COLOR_RGB2BGR)
        padded=np.full((480,640,3),255,np.uint8)
        padded[160:320,240:400]=cv2.resize(native,(160,160))
        for suffix,image in (("native",native),("padded",padded)):
            label="stop_"+font.stem+"_"+suffix
            result,items=predictions(model,image,confidence)
            summary[label]=items
            cv2.imwrite(str(output/(label+"_raw.jpg")),result.plot())
            cv2.imwrite(str(output/(label+".jpg")),annotate(image,suppress_duplicate_signs(items)))
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--weights",required=True,type=Path)
    parser.add_argument("--data",type=Path,default=Path("/workspace/traffic-sign-training/dataset"))
    parser.add_argument("--output",type=Path,default=Path("/workspace/traffic-sign-training/release-evaluation"))
    parser.add_argument("--confidence",type=float,default=.6)
    parser.add_argument("--cards-only",action="store_true")
    parser.add_argument("--metrics",action="store_true",help="Run complete heldout synthetic test metrics, including each class")
    args=parser.parse_args()
    torch.set_num_threads(2)
    model=YOLO(str(args.weights))
    summary={"weights":str(args.weights),"confidence":args.confidence,
             "scope":"Heldout synthetic images and rendered print cards; not physical-camera validation.",
             "cards":card_checks(model,args.data,args.output/"cards",args.confidence)}
    summary["cards_postprocessed"]={name:suppress_duplicate_signs(items) for name,items in summary["cards"].items()}
    if args.metrics:
        metrics=model.val(data=str(args.data/"dataset.yaml"),split="test",imgsz=320,batch=8,
                          device="cpu",workers=0,plots=True,project=str(args.output),name="test_metrics",exist_ok=True)
        summary["test_metrics"]=metrics.results_dict
        summary["test_per_class"]={model.names[int(cls)]:dict(zip(
            ("precision","recall","mAP50","mAP50_95"),map(float,metrics.box.class_result(index))))
            for index,cls in enumerate(metrics.box.ap_class_index)}
    if not args.cards_only:
        images=sorted((args.data/"images/test").glob("*.jpg"))
        negatives=[path for path in images if not (args.data/"labels/test"/(path.stem+".txt")).read_text().strip()]
        false_positives=[]
        for path in negatives:
            result,items=predictions(model,str(path),args.confidence)
            if items:
                false_positives.append({"image":path.name,"detections":items})
                cv2.imwrite(str(args.output/("negative_"+path.name)),result.plot())
        summary["hard_negative_test"]={"images":len(negatives),"images_with_detections":len(false_positives),
                                       "false_detections":false_positives}
        summary["hard_negative_test_postprocessed"]={"images":len(negatives),
            "images_with_detections":sum(bool(suppress_duplicate_signs(row["detections"])) for row in false_positives),
            "false_detections":[{"image":row["image"],"detections":suppress_duplicate_signs(row["detections"])}
                                for row in false_positives if suppress_duplicate_signs(row["detections"])]}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
