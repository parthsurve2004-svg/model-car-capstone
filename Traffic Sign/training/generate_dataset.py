"""Reproducible synthetic printed-sign data; no external images are downloaded."""
import argparse
import json
import math
import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
NAMES = ["stop", "no_entry", "turn_left", "turn_right"]
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

def card(class_id, rng, negative=False, printable=False):
    paper = tuple([rng.randint(225, 255)] * 3) if not printable else (255, 255, 255)
    image = Image.new("RGB", (256, 256), paper)
    draw = ImageDraw.Draw(image)
    red = (rng.randint(175, 235), rng.randint(8, 50), rng.randint(12, 55))
    blue = (rng.randint(8, 40), rng.randint(50, 100), rng.randint(150, 215))
    if class_id == 0:
        polygon = [(128 + 105 * math.cos(math.pi / 8 + k * math.pi / 4),128 + 105 * math.sin(math.pi / 8 + k * math.pi / 4)) for k in range(8)]
        draw.polygon(polygon, fill=red)
        inner = [(128 + 96 * math.cos(math.pi / 8 + k * math.pi / 4),128 + 96 * math.sin(math.pi / 8 + k * math.pi / 4)) for k in range(8)]
        draw.line(inner + [inner[0]], fill="white", width=4)
        text = rng.choice(["", "GO", "SLOW"]) if negative else "STOP"
        font = ImageFont.truetype(FONT_PATH, rng.randint(51, 57))
        draw.text((128, 128), text, font=font, fill="white", anchor="mm", stroke_width=0)
    elif class_id == 1:
        draw.ellipse((24, 24, 232, 232), fill=red)
        if not negative: draw.rectangle((53, 106, 203, 150), fill="white")
    else:
        if not negative or rng.random() < 0.5: draw.ellipse((24, 24, 232, 232), fill=blue)
        if not negative or image.getpixel((128, 128)) == paper:
            arrow = [(58, 128), (117, 76), (117, 106), (199, 106),(199, 150), (117, 150), (117, 180)]
            if class_id == 3: arrow = [(256 - x, y) for x, y in arrow]
            draw.polygon(arrow, fill="black" if negative else "white")
    return np.asarray(image)

def load_frames(path):
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for index in np.linspace(0, max(0, count - 1), 32, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if ok: frames.append(cv2.resize(frame, (640, 480)))
    cap.release()
    return frames

def background(rng, frames):
    if frames and rng.random() < 0.55:
        image = rng.choice(frames).copy()
        image = cv2.convertScaleAbs(image, alpha=rng.uniform(.7, 1.2), beta=rng.randint(-20, 25))
    else:
        image = np.full((480, 640, 3), rng.randint(125, 245), dtype=np.uint8)
        for _ in range(rng.randint(4, 18)):
            x, y = rng.randrange(640), rng.randrange(480)
            color = tuple(rng.randint(80, 250) for _ in range(3))
            cv2.rectangle(image, (x, y), (min(639, x+rng.randint(20, 150)), min(479, y+rng.randint(10, 80))), color, -1)
        for _ in range(rng.randint(2, 8)):
            cv2.putText(image, rng.choice(["ROAD", "CLASSROOM", "MODEL", "12345", "NMIMS"]),(rng.randrange(500), rng.randrange(30, 460)), cv2.FONT_HERSHEY_SIMPLEX,rng.uniform(.3, .8), (70, 70, 70), 1, cv2.LINE_AA)
    return image

def paste(image, patch, rng, occupied, closeup=False):
    size = rng.randint(300, 420) if closeup else rng.randint(65, 235)
    for _ in range(40):
        cx = rng.randint(size//2+8, 631-size//2)
        cy = rng.randint(size//2+8, 471-size//2)
        if all(abs(cx-x) > (size+s)*.48 or abs(cy-y) > (size+s)*.48 for x,y,s in occupied): break
    else: return None
    occupied.append((cx, cy, size))
    angle = rng.uniform(-.06, .06) if closeup else rng.uniform(-.22, .22)
    corners = []
    for x,y in [(-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5)]:
        xx = (x*math.cos(angle)-y*math.sin(angle))*size
        yy = (x*math.sin(angle)+y*math.cos(angle))*size
        corners.append((cx+xx+rng.uniform(-.07,.07)*size,cy+yy+rng.uniform(-.07,.07)*size))
    transform = cv2.getPerspectiveTransform(np.float32([[0,0],[255,0],[255,255],[0,255]]),np.float32(corners))
    warped = cv2.warpPerspective(cv2.cvtColor(patch,cv2.COLOR_RGB2BGR),transform,(640,480))
    mask = cv2.warpPerspective(np.full((256,256),255,np.uint8),transform,(640,480))
    shadow = cv2.GaussianBlur(np.roll(mask, (4,4), axis=(0,1)), (15,15), 0)/255.0
    image[:] = np.uint8(image.astype(float)*(1-shadow[...,None]*rng.uniform(.1,.25)))
    alpha = mask.astype(float)/255.0
    image[:] = np.uint8(warped*alpha[...,None]+image*(1-alpha[...,None]))
    sign_corners = cv2.perspectiveTransform(np.float32([[[24,24],[232,24],[232,232],[24,232]]]),transform)[0]
    x1,y1 = sign_corners.min(axis=0)
    x2,y2 = sign_corners.max(axis=0)
    return [max(0,x1),max(0,y1),min(640,x2),min(480,y2)]

def generate(output, repo, counts, seed):
    output.mkdir(parents=True, exist_ok=True)
    # Labels are regenerated; never reuse an earlier Ultralytics label cache.
    for split in ("train","val","test"):
        (output/"labels"/(split+".cache")).unlink(missing_ok=True)
    videos = {"train":repo/"benchmarking/test_videos/dashcam_test.mp4","val":repo/"benchmarking/test_videos/testvideovid1.mp4","test":repo/"pothole_test_2.mp4"}
    manifest = {"seed":seed,"classes":NAMES,"counts":{},"background_video_by_split":{k:str(v) for k,v in videos.items()},"limitations":"Synthetic sign overlays and heldout video/procedural backgrounds only. Not physical-camera evaluation."}
    for split,count in zip(["train","val","test"],counts):
        rng = random.Random(seed + {"train":0,"val":100000,"test":200000}[split])
        np_rng = np.random.default_rng(seed+{"train":0,"val":100000,"test":200000}[split])
        frames = load_frames(videos[split])
        image_dir, label_dir = output/"images"/split, output/"labels"/split
        image_dir.mkdir(parents=True,exist_ok=True)
        label_dir.mkdir(parents=True,exist_ok=True)
        positives=0
        for index in range(count):
            image=background(rng,frames)
            labels=[]
            occupied=[]
            negative_frame=index%3==0
            closeup=index%5==1
            for _ in range(1 if closeup else rng.randint(1,2)):
                cls=0 if negative_frame and rng.random()<.6 else rng.randrange(4)
                bounds=paste(image,card(cls,rng,negative=negative_frame),rng,occupied,closeup=closeup)
                if bounds is not None and not negative_frame:
                    x1,y1,x2,y2=bounds
                    labels.append(f"{cls} {(x1+x2)/1280:.6f} {(y1+y2)/960:.6f} {(x2-x1)/640:.6f} {(y2-y1)/480:.6f}")
            if rng.random()<.4: image=cv2.GaussianBlur(image,(3,3),rng.uniform(.3,1.0))
            image=np.uint8(np.clip(image.astype(float)+np_rng.normal(0,rng.uniform(0,3),image.shape),0,255))
            cv2.imwrite(str(image_dir/f"{index:05}.jpg"),image,[cv2.IMWRITE_JPEG_QUALITY,rng.randint(65,95)])
            (label_dir/f"{index:05}.txt").write_text("\n".join(labels))
            positives+=len(labels)
        manifest["counts"][split]={"images":count,"sign_instances":positives,"negative_images":(count+2)//3}
        print(split,manifest["counts"][split],flush=True)
    (output/"dataset.yaml").write_text(f"path: {output}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"+"".join(f"  {i}: {name}\n" for i,name in enumerate(NAMES)))
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    printable=output/"printable_cards"
    printable.mkdir(exist_ok=True)
    for cls,name in enumerate(NAMES):
        Image.fromarray(card(cls,random.Random(seed+cls),printable=True)).resize((1024,1024)).save(printable/f"{name}.png")

def add_full_card_samples(output, seed):
    """Include card-only views so native printable images are not out of scale."""
    manifest=json.loads((output/"manifest.json").read_text())
    for split in ("train","val","test"):
        rng=random.Random(seed+{"train":300000,"val":400000,"test":500000}[split])
        count=manifest["counts"][split]["images"]
        native_count=0
        for index in range(2,count,11):
            cls=rng.randrange(4)
            negative=index%3==0
            image=card(cls,rng,negative=negative)
            cv2.imwrite(str(output/"images"/split/f"{index:05}.jpg"),cv2.cvtColor(image,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,90])
            label="" if negative else f"{cls} 0.500000 0.500000 0.812500 0.812500"
            (output/"labels"/split/f"{index:05}.txt").write_text(label)
            native_count+=1
        manifest["counts"][split]["native_card_images"]=native_count
        manifest["counts"][split]["sign_instances"]=sum(len(p.read_text().splitlines()) for p in (output/"labels"/split).glob("*.txt"))
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path("/workspace/traffic-sign-training/dataset"))
    parser.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument("--seed",type=int,default=20261030)
    args=parser.parse_args()
    generate(args.output.resolve(),args.repo,(768,128,128),args.seed)
    add_full_card_samples(args.output.resolve(),args.seed)
