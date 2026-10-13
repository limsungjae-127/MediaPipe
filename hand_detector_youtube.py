#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp
import yt_dlp

try:
    import mediapipe.python.solutions.hands as mp_hands
    import mediapipe.python.solutions.drawing_utils as mp_drawing
    import mediapipe.python.solutions.drawing_styles as mp_styles
except ImportError:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

def get_youtube_stream(url):
    ydl_opts = {
        'quiet': True,
        'format': 'best[ext=mp4]/best',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info['url']

# 정상 재생 가능한 유튜브 영상 주소 지정
youtube_url = "https://www.youtube.com/watch?v=gA7GqciN160"

print("📹 YouTube 스트림 URL을 가져오는 중...")
try:
    video_stream_url = get_youtube_stream(youtube_url)
    print("✔ 스트림 URL 추출 완료")
except Exception as e:
    print(f"❌ 유튜브 스트림 추출 실패: {e}")
    exit()

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(video_stream_url)

if not cap.isOpened():
    print("❌ YouTube 영상을 열 수 없습니다.")
    exit()

print("🎥 YouTube 스트림 재생 시작 - ESC를 눌러 종료합니다.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("⚠️ 영상 스트림이 끝났거나 프레임을 읽지 못했습니다.")
        break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

    cv2.imshow('🖐️ MediaPipe Hand Detector (YouTube)', image)
    if cv2.waitKey(5) & 0xFF == 27:
        print("👋 종료합니다.")
        break

cap.release()
cv2.destroyAllWindows()
