import cv2
import numpy as np
import onnxruntime as ort

print("Initializing Pentium-Safe ONNX Runtime Engine (v1.17.0)...")

# 1. Load the ONNX model using our working runtime engine
# This completely bypasses OpenCV's broken shape_utils module!
session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])

# Get the exact input layer metadata expected by the model
input_meta = session.get_inputs()[0]
input_name = input_meta.name

# Custom dataset class labels
classes = ["100_Rupees", "200_Rupees", "20_Rupees", "500_Rupees", "50_Rupees"]

# 2. Access your hardware webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam setup.")
    exit()

print("\n🚀 System running beautifully via ONNX Runtime!")
print("Hold your currency note up to the camera. Press 'q' inside the window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h_img, w_img, _ = frame.shape

    # 3. Pre-process the webcam frame to a strict 640x640 shape for YOLOv8
    # Resize, scale pixel values to 0.0-1.0, and change HWC layout to CHW layout
    img_resized = cv2.resize(frame, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    blob = np.expand_dims(
        img_transposed, axis=0
    )  # Add batch dimension -> [1, 3, 640, 640]

    # 4. Run the math pass cleanly using the onnxruntime session
    outputs = session.run(None, {input_name: blob})

    # Isolate batch dimension and transpose [1, 9, 8400] -> [8400, 9]
    predictions = np.squeeze(outputs[0]).T

    # 5. Parse bounding box coordinates and object classifications
    for row in predictions:
        scores = row[4:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.6:  # 60% confidence filter
            x_center, y_center, width, height = row[0:4]

            # Re-scale bounding box dimensions back up to match your screen resolution
            x1 = int((x_center - width / 2) * w_img / 640)
            y1 = int((y_center - height / 2) * h_img / 640)
            x2 = int((x_center + width / 2) * w_img / 640)
            y2 = int((y_center + height / 2) * h_img / 640)

            # Draw tracking rectangle box over the detected currency note
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 118), 2)

            # Add text label with percentage
            label = f"{classes[class_id]}: {confidence * 100:.1f}%"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 230, 118),
                2,
            )

    # 6. Display visual camera tracking stream
    cv2.imshow("Pentium Local Currency Tracking", frame)

    # Exit out when 'q' key is pressed inside the GUI display window
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("\nStream closed down safely.")
