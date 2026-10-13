#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp
import yt_dlp

def get_youtube_stream(url):
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'format': 'b/bestvideo+bestaudio/best',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        
        if 'url' in info:
            return info['url']
        elif 'formats' in info and len(info['formats']) > 0:
            valid_formats = [f for f in info['formats'] if f.get('url')]
            if valid_formats:
                return valid_formats[-1]['url']
        
        raise KeyError("스트림 URL을 찾을 수 없습니다.")

# 유튜브 쇼츠 URL
youtube_url = "https://youtu.be/ZLGZlJsxtCM"

print("📹 YouTube 스트림 URL을 가져오는 중...")
try:
    video_stream_url = get_youtube_stream(youtube_url)
    print("✔ 스트림 URL 추출 완료")
except Exception as e:
    print(f"❌ 유튜브 스트림 추출 실패: {e}")
    exit()

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(video_stream_url)

if not cap.isOpened():
    print("❌ YouTube 영상을 열 수 없습니다.")
    exit()

print("🎥 YouTube 스트림 재생 시작 - ESC를 눌러 종료합니다.")

# model_selection=1 (전신/원거리/다양한 각도 모델), min_detection_confidence=0.3 (가려짐 대비 감도 향상)
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("⚠️ 영상 스트림이 끝났거나 프레임을 읽지 못했습니다.")
            break

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

        cv2.imshow('MediaPipe Face Detection (YouTube)', image)
        if cv2.waitKey(5) & 0xFF == 27:
            print("👋 종료합니다.")
            break

cap.release()
cv2.destroyAllWindows()