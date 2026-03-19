import time
import numpy as np
from ultralytics import YOLO

# 1. 서버 구동 시 모델 로드 및 웜업(예열)
# 주의: 백엔드 A가 서버를 돌릴 때 이 경로에 best.pt를 둬야 합니다.
MODEL_PATH = "weights/best.pt" 

print("⏳ AI 모델을 메모리에 올리고 예열을 시작합니다...")
model = YOLO(MODEL_PATH)

# 더미 데이터로 1회 헛돌게 하여 첫 추론 지연(Cold Start) 방지
dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
model.predict(source=dummy_frame, imgsz=640, verbose=False)
print("✅ AI 모델 로드 및 예열 완료 (준실시간 추론 준비됨)")

# 2. 백엔드 로직에 넘길 최소 신뢰도 (40% 이상만 포트홀로 취급)
CONF_THRESHOLD = 0.4 

def run_inference(frame: np.ndarray, frame_timestamp: float, device_id: str) -> dict:
    """
    [입력]
    - frame: 백엔드 A가 OpenCV로 읽어서 넘겨준 이미지 배열
    - frame_timestamp: 스마트폰이 찍은 시간
    - device_id: 스마트폰 기기 번호
    
    [출력]
    - 약속된 규격의 Dictionary (나중에 백엔드가 JSON으로 변환)
    """
    # 추론 속도 측정을 위해 시작 시간을 기록.
    start_time = time.time()
    
    # YOLO 모델 추론 (imgsz=640 고정으로 연산량 최소화, 준실시간성 확보)
    results = model.predict(source=frame, conf=CONF_THRESHOLD, imgsz=640, verbose=False)
    
    # 걸린 시간 계산 (밀리초 단위)
    inference_time_ms = round((time.time() - start_time) * 1000, 2)
    
    detections = []
    
    # 프레임에서 찾은 포트홀 개수만큼 반복하여 리스트에 담음.
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist() # Bbox 픽셀 좌표
        conf = float(box.conf[0])             # 신뢰도 확률
        
        detections.append({
            "class": "pothole",
            "confidence": round(conf, 2),
            "bbox": [int(x1), int(y1), int(x2), int(y2)]
        })
        
    # 백엔드 A와 약속한 최종 출력 규격으로 조립.
    output_data = {
        "frame_timestamp": frame_timestamp,
        "device_id": device_id,
        "inference_time_ms": inference_time_ms,
        "detections": detections
    }
    
    return output_data
