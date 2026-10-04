import cv2
import os

video_path = r"C:\Users\Dell\Downloads\mujhe_isa_photo_ko_lively_anim.mp4"
output_dir = r".\video_test_frames"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

indices = [int(i * total_frames / 16) for i in range(16)]

saved = 0

for frame_index in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = cap.read()

    if success:
        output_path = os.path.join(output_dir, f"frame_{saved:02d}.jpg")
        cv2.imwrite(output_path, frame)
        saved += 1

cap.release()

print("Total video frames:", total_frames)
print("Frames requested:", 16)
print("Frames saved:", saved)
print("Output folder:", os.path.abspath(output_dir))
