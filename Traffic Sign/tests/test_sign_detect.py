import json
import cv2
import numpy as np
import detect

class Predictor:
    def __init__(self,path,rate_hz,confidence,clock):
        self.clock=clock
        self.interval=1/rate_hz
        self.next=float('-inf')
    def poll(self,frame):
        if self.clock()<self.next:
            return None
        self.next=self.clock()+self.interval
        return [{'class_name':'stop','confidence':.95,'bbox':[15.,15.,65.,65.],
                 'type':'traffic_sign'}]


def test_motor_free_image_output_and_json(tmp_path):
    source=tmp_path/'input.png'
    cv2.imwrite(str(source),np.zeros((100,100,3),np.uint8))
    model=tmp_path/'custom.pt'
    model.touch()
    output=tmp_path/'results'
    result=detect.run(source,model=model,output=output,detector_factory=Predictor)
    assert result['frames'][0]['fresh_prediction']
    assert result['frames'][0]['detections'][0]['class_name']=='stop'
    assert result['frames'][0]['detections'][0]['actionable']
    assert (output/'annotated.jpg').is_file()
    assert np.any(cv2.imread(str(output/'annotated.jpg')))
    assert json.loads((output/'detections.json').read_text())==result


def test_prerecorded_video_sampling_uses_media_time_not_decode_speed(tmp_path,monkeypatch):
    source=tmp_path/'input.mp4'
    source.touch()
    model=tmp_path/'custom.pt'
    model.touch()
    class Capture:
        index=0
        released=False
        def isOpened(self): return True
        def get(self,key): return 8.
        def read(self):
            self.index+=1
            return (True,np.zeros((100,100,3),np.uint8)) if self.index<=8 else (False,None)
        def release(self): self.released=True
    class Writer:
        released=False
        def isOpened(self): return True
        def write(self,frame): pass
        def release(self): self.released=True
    capture,writer=Capture(),Writer()
    monkeypatch.setattr(detect.cv2,'VideoCapture',lambda *args:capture)
    monkeypatch.setattr(detect.cv2,'VideoWriter',lambda *args:writer)
    result=detect.run(source,model=model,output=tmp_path/'results',detector_factory=Predictor)
    assert sum(row['fresh_prediction'] for row in result['frames'])==4
    assert len(result['frames'])==8
    assert capture.released and writer.released
