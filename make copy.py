import cv2

src = r".\YOLOX_outputs\yolox_x_mix_det\track_vis\2026_08_26_12_51_16\partial_to_frame_182.avi"
out = r".\YOLOX_outputs\yolox_x_mix_det\track_vis\2026_08_26_12_51_16\palace_first_178_frames.avi"

cap = cv2.VideoCapture(src)

if not cap.isOpened():
    raise RuntimeError("Could not open partial video.")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    out,
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps if fps > 0 else 30,
    (width, height)
)

count = 0

while count < 178:
    ret, frame = cap.read()

    if not ret:
        break

    writer.write(frame)
    count += 1

cap.release()
writer.release()

print("Created:", out)
print("Frames written:", count)