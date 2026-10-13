#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import mediapipe as mp

mp_selfie = mp.solutions.selfie_segmentation
segmenter = mp_selfie.SelfieSegmentation(model_selection=1)

cap = cv2.VideoCapture("face.mp4")

print("📷 Selfie Segmentation 시작 — ESC를 눌러 종료합니다.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("⚠️ 프레임을 읽지 못했습니다. face.mp4 파일을 확인하세요.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = segmenter.process(rgb)

    mask = (result.segmentation_mask > 0.5).astype(np.uint8)
    blur = cv2.GaussianBlur(frame, (31, 31), 0)

    output = frame * mask[..., None] + blur * (1 - mask[..., None])
    output = output.astype(np.uint8)

    cv2.imshow("🪞 Selfie Segmentation (ESC to quit)", output)
    if cv2.waitKey(1) & 0xFF == 27:
        print("👋 종료합니다.")
        break

cap.release()
cv2.destroyAllWindows()
