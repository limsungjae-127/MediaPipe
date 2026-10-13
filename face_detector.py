#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture("face.mp4")

print("📷 얼굴 감지 시작 — ESC를 눌러 종료합니다.")

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("⚠️ 영상 프레임을 읽지 못했습니다. face.mp4 파일을 확인하세요.")
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

        cv2.imshow('MediaPipe Face Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:
            print("👋 종료합니다.")
            break

cap.release()
cv2.destroyAllWindows()
