import cv2
import numpy as np
import onnxruntime as ort

print("Initializing Highly Optimized Pentium ONNX Engine...")

session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

classes = ["100_Rupees", "200_Rupees", "20_Rupees", "500_Rupees", "50_Rupees"]
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam setup.")
    exit()

# --- SPEED OPTIMIZATION VARIABLES ---
frame_count = 0
skip_frames = 4  # Run AI math only every 4th frame. Increase this (e.g., 5 or 6) if it still lags!
saved_boxes = []  # Stores boxes to draw during skipped frames

print("\n🚀 Optimization active! Lag reduction system engaged.")
print("Press 'q' inside the window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h_img, w_img, _ = frame.shape
    frame_count += 1

    # ONLY RUN HEAVY AI MATH ON DESIGNATED FRAMES
    if frame_count % skip_frames == 0:
        saved_boxes = []  # Clear old tracking boxes

        # Downsample image shape quickly for processing speed
        img_resized = cv2.resize(frame, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        blob = np.expand_dims(img_transposed, axis=0)

        # Run model inference
        outputs = session.run(None, {input_name: blob})
        predictions = np.squeeze(outputs[0]).T

        # High threshold filter (0.65) to quickly ignore empty background blocks
        for row in predictions:
            scores = row[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.65:
                x_center, y_center, width, height = row[0:4]

                x1 = int((x_center - width / 2) * w_img / 640)
                y1 = int((y_center - height / 2) * h_img / 640)
                x2 = int((x_center + width / 2) * w_img / 640)
                y2 = int((y_center + height / 2) * h_img / 640)

                # Save coordinates to render smoothly over skipped frames
                saved_boxes.append((x1, y1, x2, y2, class_id, confidence))

    # DRAW THE SAVED BOXES ON EVERY FRAME (ZERO LAG COST)
    for box in saved_boxes:
        x1, y1, x2, y2, class_id, confidence = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 118), 2)
        label = f"{classes[class_id]}: {confidence * 100:.1f}%"
        cv2.putText(
            frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 118), 2
        )

    # Display camera window
    cv2.imshow("Pentium Local Currency Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("\nStream closed safely.")
