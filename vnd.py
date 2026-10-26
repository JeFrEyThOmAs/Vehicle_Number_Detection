# import cv2
# import easyocr
# from ultralytics import YOLO

# # 1. Initialize YOLOv8 (Vehicle Detection) and EasyOCR (Text Reading)
# print("Loading models... Please wait.")
# coco_model = YOLO("yolov8n.pt")
# reader = easyocr.Reader(["en"], gpu=False)  # Set gpu=True if running with CUDA

# # COCO Class IDs for vehicles: 2 = Car, 3 = Motorcycle, 5 = Bus, 7 = Truck
# VEHICLE_CLASSES = [2, 3, 5, 7]

# # 2. Initialize video stream (Use 0 for live webcam or 'car_video.mp4' for a video file)
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Could not open video source.")
#     exit()

# print("Starting Number Plate Detection. Press 'q' to quit.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Video stream finished or frame unreadable.")
#         break

#     # 3. Detect vehicles using YOLOv8
#     results = coco_model(frame, verbose=False)[0]

#     for box in results.boxes:
#         class_id = int(box.cls[0])
#         confidence = float(box.conf[0])

#         # Filter detections for vehicles only (confidence > 50%)
#         if class_id in VEHICLE_CLASSES and confidence > 0.5:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])

#             # Draw a green bounding box around the vehicle
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(
#                 frame,
#                 f"Vehicle {confidence:.2f}",
#                 (x1, max(y1 - 10, 20)),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.5,
#                 (0, 255, 0),
#                 2,
#             )

#             # 4. Crop the vehicle area to inspect for license plate text
#             vehicle_crop = frame[y1:y2, x1:x2]

#             if vehicle_crop.size > 0:
#                 # 5. Run EasyOCR on the cropped vehicle image
#                 ocr_results = reader.readtext(vehicle_crop)

#                 for bbox, text, prob in ocr_results:
#                     # Clean extracted text (keep alphanumeric characters only)
#                     clean_text = "".join(e for e in text if e.isalnum()).upper()

#                     # Filter for realistic plate length and confidence score
#                     if prob > 0.35 and len(clean_text) >= 4:
#                         # Convert OCR coordinates back to full frame dimensions
#                         (top_left, _, bottom_right, _) = bbox
#                         px1 = int(top_left[0]) + x1
#                         py1 = int(top_left[1]) + y1
#                         px2 = int(bottom_right[0]) + x1
#                         py2 = int(bottom_right[1]) + y1

#                         # Draw a red box around the license plate and overlay the text
#                         cv2.rectangle(
#                             frame, (px1, py1), (px2, py2), (0, 0, 255), 2
#                         )
#                         cv2.putText(
#                             frame,
#                             f"Plate: {clean_text}",
#                             (px1, max(py1 - 10, 20)),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             0.7,
#                             (0, 0, 255),
#                             2,
#                         )

#     # 6. Display output window
#     cv2.imshow("Vehicle & License Plate Detection", frame)

#     # Press 'q' to stop
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import easyocr
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# --------------------------------------------------
# 1. LOAD MODELS
# --------------------------------------------------

print("Loading license plate detection model...")

model_path = hf_hub_download(
    repo_id="66777yui/yolov8-license-plate-detection",
    filename="best.pt"
)

plate_model = YOLO(model_path)

print("Loading EasyOCR...")

reader = easyocr.Reader(["en"], gpu=False)

print("Models loaded successfully!")

# --------------------------------------------------
# 2. OPEN WEBCAM
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit(1)

print("Vehicle Number Plate Detection Started!")
print("Press 'q' to quit.")

frame_count = 0

# Store previously recognized text to reduce OCR calls.
last_texts = {}

# --------------------------------------------------
# 3. PROCESS VIDEO
# --------------------------------------------------

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    frame_count += 1

    # Detect license plates directly.
    results = plate_model(
        frame,
        conf=0.35,
        verbose=False
    )[0]

    current_texts = {}

    # --------------------------------------------------
    # 4. PROCESS EACH DETECTED PLATE
    # --------------------------------------------------

    for index, box in enumerate(results.boxes):

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        detection_confidence = float(box.conf[0])

        # Keep coordinates inside the image.
        height, width = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        # Crop from the original, unannotated frame.
        plate_crop = frame[y1:y2, x1:x2].copy()

        if plate_crop.size == 0:
            continue

        plate_text = ""

        # --------------------------------------------------
        # 5. RUN OCR EVERY 3 FRAMES
        # --------------------------------------------------

        cache_key = (x1 // 30, y1 // 30, x2 // 30, y2 // 30)

        if frame_count % 3 == 0:
            # Enlarge small plate images for OCR.
            plate_crop = cv2.resize(
                plate_crop,
                None,
                fx=2,
                fy=2,
                interpolation=cv2.INTER_CUBIC
            )

            # Read text from the cropped plate.
            ocr_results = reader.readtext(
                plate_crop,
                detail=1
            )

            best_text = ""
            best_confidence = 0.0

            for _, text, confidence in ocr_results:

                # Keep letters and digits only.
                cleaned = "".join(
                    char for char in text
                    if char.isalnum()
                ).upper()

                if len(cleaned) >= 4 and confidence > best_confidence:
                    best_text = cleaned
                    best_confidence = confidence

            if best_confidence >= 0.30:
                plate_text = best_text

            current_texts[cache_key] = plate_text

        else:
            plate_text = last_texts.get(cache_key, "")

        # --------------------------------------------------
        # 6. DRAW DETECTION BOX AND TEXT
        # --------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        if plate_text:
            label = f"{plate_text} ({detection_confidence:.2f})"
        else:
            label = f"Plate detected ({detection_confidence:.2f})"

        text_y = max(y1 - 10, 25)

        cv2.putText(
            frame,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

    # Update the OCR cache.
    if frame_count % 3 == 0:
        last_texts = current_texts

    # --------------------------------------------------
    # 7. DISPLAY OUTPUT
    # --------------------------------------------------

    cv2.imshow(
        "Vehicle Number Plate Recognition",
        frame
    )

    # Press q to exit.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --------------------------------------------------
# 8. CLEANUP
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()

print("Number plate detection stopped.")