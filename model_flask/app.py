# Flask import
from flask import Flask, render_template, Response, stream_with_context
from flask import g

# Flask socks
from flask_sock import Sock
from simple_websocket import Server as WSServer

# other import
import cv2
import numpy as np
import torch
import cmapy
import json
import platform
import time
from threading import Thread
from queue import Queue

# Singleton: 클래스 선언시, 인스턴스가 하나만 생성되도록 함
class Singleton(type):
    """ 
    Simple Singleton that keep only one value for all instances
    """

    def __init__(cls, name, bases, dic):
        super(Singleton, cls).__init__(name, bases, dic)
        cls.instance = None

    def __call__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super(Singleton, cls).__call__(*args, **kwargs)
        return cls.instance

# Models: 모델을 로드하고 저장하는 클래스
class Models(metaclass=Singleton):
    def __init__(self, equip_model_path = 'models/equip/best.pt', handmotion_model_path = 'models/handmotion/best.pt'):
        # 연산 device 설정
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # torch hub를 사용한 model 로드
        self.equip_model = torch.hub.load('ultralytics/yolov5', 'custom', path=equip_model_path)
        self.handmotion_model = torch.hub.load('ultralytics/yolov5', 'custom', path=handmotion_model_path)
        
        # 해당 모델을 사용할 device로 옮김
        self.equip_model.to(self.device)
        self.handmotion_model.to(self.device)
    
    # 이미지에서 Object Detection을 수행
    def detect(self, img, mhi_img, size = 640):
        ret = []
        
        # 이미지 predict
        results = {
            'handmotion': self.handmotion_model(mhi_img, size),
            'equip': self.equip_model(img, size)
        }

        # 한 return 값으로 합침
        for model_name, result in results.items():
            for i in range(len(result.xyxy[0])):
                r = result.pandas().xyxy[0].iloc[i, :].values.tolist()
                x1, y1, x2, y2, confidence, cls_id, cls_name = r
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                confidence, cls_id = float(confidence), int(cls_id)

                ret.append({
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'confidence': confidence,
                    'cls_id': cls_id,
                    'cls_name': cls_name,
                    'model': model_name,
                })

        return ret


# 스트리밍용 클래스
class Streamer():

    def __init__(self, device_id, pos=15, cmap = 'nipy_spectral', confidence = 0.5):
        # 기본 설정
        self.device = None
        self.device_id = device_id
        self.started = False
        self.thread = None
        
        # 모델 설정
        self.confidence = confidence
        
        # MHI용 설정
        self.pos = pos
        self.cmap = cmapy.cmap(cmap)

        # 배경 제거 모델
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        
        # 모델
        self.models = Models()
        
        # frame queue
        self.queue_size = 64
        self.frame_queue = Queue(maxsize=self.queue_size)
        self.mhi_queue = Queue(maxsize=self.queue_size)
        self.boxed_queue = Queue(maxsize=self.queue_size)
        
        # ws
        self.ws_client_list = []

    # 실행
    # Thread를 사용하여, 실시간으로 frame을 받아와 저장
    def run(self) :
        self.stop()
        
        if self.device is None:
            if platform.system() == 'Windows' :        
                self.device = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
            else:
                self.device = cv2.VideoCapture(self.device_id)

            #self.fps = int(1000/self.device.get(cv2.CAP_PROP_FPS))
            self.width = int(self.device.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.device.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.mhi_history = np.zeros((self.height, self.width), dtype=np.int16)
        
        # Thread 시작
        if self.thread is None :
            self.thread = Thread(target=self.update, args=())
            self.thread.daemon = True
            self.thread.start()
        
        self.started = True
    
    # Thread 및 Device 정지
    def stop(self):
        self.started = False
        
        if self.device is not None :
            self.device.release()
            self.clear()
    
    # Queue 정리 (최소 단위 및 해당하는 queue만)
    def clear_queue(self, time = None):
        if time is None:
            time = self.queue_size // 2
        
        if self.frame_queue.qsize() > time:
            self.clear_frame()
        
        if self.mhi_queue.qsize() > time:
            self.clear_mhi()
        
        if self.boxed_queue.qsize() > time:
            self.clear_boxed()

    # Queue 정리
    def clear(self):
        self.clear_frame()
        self.clear_mhi()
        self.clear_boxed()

    def clear_frame(self):
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
    
    def clear_mhi(self):
        with self.mhi_queue.mutex:
            self.mhi_queue.queue.clear()
    
    def clear_boxed(self):
        with self.boxed_queue.mutex:
            self.boxed_queue.queue.clear()
            
    def drop_queue(self, time = None):
        if time is None:
            time = self.queue_size // 2

        if self.frame_queue.qsize() > time:
            self.drop_frame_queue(time)
        
        if self.mhi_queue.qsize() > time:
            self.drop_mhi_queue(time)
        
        if self.boxed_queue.qsize() > time:
            self.drop_boxed_queue(time)

        
    def drop_frame_queue(self, time):
        for _ in range(time):
            self.frame_queue.get()

    def drop_mhi_queue(self, time):
        for _ in range(time):
            self.mhi_queue.get()
    
    def drop_boxed_queue(self, time):
        for _ in range(time):
            self.boxed_queue.get()

    # Thread 함수
    def update(self):
        while True:
            # 시작할때까지 대기
            if not self.started:
                continue

            # 임시. Queue 사이즈가 절반 이상이면 clear
            # Queue를 buffer로 써 계속 streaming을 반복하기 위함.
            self.clear_queue()

            try:
                self.frame_queue.put(self.get_frame())
                self.mhi_queue.put(self.get_mhi())
                self.detected = detected = self.detect()
                self.boxed_queue.put(self.draw_box(detected))
                self.send_detected_ws()
            except Exception as e:
                print(e)
                continue
            
    def send_detected_ws(self, detected = None):
        if detected is None:
            detected = self.detected
        
        clients = self.ws_client_list.copy()
        for client in clients:
            try:
                j = json.dumps(detected)
                client.send(j)
            except Exception as e:
                client.send(json.dumps({'error': str(e)}))
                client.close()
                self.ws_client_list.remove(client)

    # 빈 frame
    def blank(self):
        return np.ones(shape=[self.height, self.width, 3], dtype=np.uint8)
    
    # 빈 jpg
    def blank_jpg(self):
        return cv2.imencode('.jpg', self.blank())[1]

    # frame을 받아와서 queue에 저장
    def get_frame(self):
        # 진행전 전처리
        if self.device is None or self.device.isOpened() is False:
            frame = self.blank()
        else :
            ret, frame = self.device.read()
            if ret != True:
                frame = self.blank()
        
        self.frame = frame
        return frame

    # mhi stands for motion history image
    def get_mhi(self):
        # 배경 제거 마스크 계산
        fgmask = self.fgbg.apply(self.frame)

        # make 24 frame of history based on fgmask
        self.mhi_history = np.where(fgmask == 255, 255 + self.pos, self.mhi_history)
        self.mhi_history = np.where(self.mhi_history > -100, self.mhi_history - self.pos, self.mhi_history)

        # make clip for safety
        self.mhi_history = np.clip(self.mhi_history, 0, 255)
        history_frame = self.mhi_history.astype(np.uint8)

        # color map
        color_map_history = cv2.applyColorMap(history_frame, self.cmap)

        self.mhi_frame = color_map_history
        return color_map_history
    
    # 감지   
    def detect(self):
        self.detected = self.models.detect(self.frame, self.mhi_frame)
        return self.detected
    
    # bbox 그리기
    def draw_box(self, detect = None):
        if detect is None:
            detect = self.detect()
        f = self.frame.copy()
        
        for d in detect:
            name = f'{d["model"]}_{d["cls_name"]}, {d["confidence"]:.2f}'
            
            x1, y1, x2, y2 = d['x1'], d['y1'], d['x2'], d['y2']
            cv2.rectangle(f, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(f, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        self.boxed_frame = f
        return f
    
    # Queue에서 frame을 받아옴
    def read_frame(self):
        return self.frame_queue.get(block=True, timeout = 20)
    
    # Queue에서 mhi frame를 받아옴
    def read_mhi(self):
        return self.mhi_queue.get(block=True, timeout = 20)
    
    # Queue에서 boxed frame를 받아옴
    def read_boxed(self):
        return self.boxed_queue.get(block=True, timeout = 20)
    
    # jpg로 변환하여 stream에 맞는 형태로 변환
    def _gen(self, f, name):
        ret, buffer = cv2.imencode('.jpg', f)
        if ret != True:
            buffer = self.blank_jpg()
        frame = buffer.tobytes()
        return (b'--' + name + b'\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    def gen_frame(self):
        try:
            d = self.read_frame()
        except Exception as e:
            d = self.blank()
        return self._gen(d, b'image_frame')
    
    def gen_mhi(self):
        try:
            d = self.read_mhi()
        except Exception as e:
            d = self.blank()
        return self._gen(d, b'image_mhi')

    def gen_boxed(self):
        try:
            d = self.read_boxed()
        except Exception as e:
            d = self.blank()
        return self._gen(d, b'image_boxed')

# Stream 을 맞게 yield 하는 함수
def stream_gen(streamer, target = 'boxed'):
    try:      
        while True:
            if streamer.started:
                if target == 'boxed':
                    yield streamer.gen_boxed()
                elif target == 'mhi':
                    yield streamer.gen_mhi()
                elif target == 'frame':
                    yield streamer.gen_frame()
                else:
                    raise Exception('Invalid target')
            else:
                raise Exception('Streamer stopped')
    except Exception as e:
        streamer.stop()
        return e

# 중복 방지 및 context 유지를 위하여 flask에서 사용하는 global 변수를 사용
def get_streamer(device_id, pos=15, cmap = 'nipy_spectral', confidence = 0.5) -> Streamer:
    n = f'streamer_{device_id}'
    s = getattr(g, n, None)
    if s is None:
        s = Streamer(device_id, pos, cmap, confidence)
        setattr(g, n, s)
        s.run()
    return s

# flask app
app = Flask(__name__)
sock = Sock(app)


# box를 그린 frame을 stream
@app.route('/feed/video/<int:device_id>', methods=['GET'])
def video_feed(device_id):    
    try:
        streamer = get_streamer(device_id)

        return Response(stream_with_context(stream_gen(streamer, 'boxed')),
                    mimetype='multipart/x-mixed-replace; boundary=image_boxed')
    
    except Exception as e:
        print(e)
        return Response(
            "Unknown Server Error",
            status=500,
        )

# json을 stream (via WebSocket)
@sock.route('/feed/json/<int:device_id>')
def json_feed(ws: WSServer, device_id):
    try:
        streamer = get_streamer(device_id)

        streamer.ws_client_list.append(ws)
        while True:
            data = ws.receive()
            if data == 'stop':
                break
        streamer.ws_client_list.remove(ws)
    except Exception as e:
        print(e)
        ws.send(e)
    finally:
        ws.close()

# Flask 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)  # debug=True
