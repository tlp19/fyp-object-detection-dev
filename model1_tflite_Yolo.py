# import the opencv library
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time

# define a video capture object
# cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cam.set(cv2.CAP_PROP_FPS, 5)
cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)

# MODEL_PATH = "./models/model1-train2/best_saved_model/best_float16.tflite"     # Good - 704
# MODEL_PATH = "./models/model1-train2/best_saved_model/best_int8.tflite"        # Good - 704
# MODEL_PATH = "./models/model1-train1/best_saved_model/best_float16.tflite"     # Broken
# MODEL_PATH = "./models/model1-train4/best_float16.tflite"                      # Broken 
# MODEL_PATH = "./models/model1-train5/best_float16.tflite"                      # Poor - 640
MODEL_PATH = "./models/model1-train6/best_float16.tflite"                      # Okay - 320

CONFIDENCE_THRESHOLD = 0.2

CLASS_LABELS = ["cauli", "other"]


colors = np.random.uniform(0, 255, size=(len(CLASS_LABELS), 3)) / 255.0

# Load TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
# Get input and output tensor details.
input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']
output_details = interpreter.get_output_details()
print("input shape:", input_shape)
print("input type:", input_details[0]['dtype'])

running_fps = 0
running_inf_time = 0
iterations = 0


def detect(interpreter, data, top_k=1):
    # Set input of the network
    #input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], data)

    # Run the network
    interpreter.invoke()

    # Get output of the network
    #output_details = interpreter.get_output_details()
    output = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.

    # Take the transpose to have 1 detection per row
    output = np.transpose(output)

    # Analyze outputs
    boxes = []
    scores = []
    class_ids = []

    for row in output:
        class_scores = row[4:]
        _, max_score, _, max_class = cv2.minMaxLoc(class_scores)
        if max_score >= CONFIDENCE_THRESHOLD:
            left = row[0] - row[2] / 2.0
            top = row[1] - row[3] / 2.0
            right = row[0] + row[2] / 2.0
            bottom = row[1] + row[3] / 2.0
            box = (left, top, right, bottom)
            boxes.append(box)
            scores.append(max_score)
            class_ids.append(max_class[1]) # max_class is a tuple, take the column idx
    
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, 0.45)

    detections = []
    for i in range(len(result_boxes)):
        idx = result_boxes[i]
        box = boxes[idx]
        detection = {
            "class_id": class_ids[idx],
            "class_name": CLASS_LABELS[class_ids[idx]],
            "confidence": scores[idx],
            "box": box,
        }
        detections.append(detection)

    return detections


def draw_detection(image, detection):
    text = f"{detection['class_name']} {detection['confidence']:.2f}"
    color = colors[detection['class_id']]
    bbox = detection['box']
    start = (round(bbox[0]), round(bbox[1]))
    end = (round(bbox[2]), round(bbox[3]))
    cv2.rectangle(image, start, end, color, 2)
    cv2.putText(image, text, (start[0], start[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


while(True):
    # Capture the video frame by frame
    # First, skip the buffered image
    success, frame = cam.read()
    # Then, capture a fresh one
    success, frame = cam.read()

    if not success:
        print("Camera could not be read")
        break

    else:
        iterations += 1
        print(f"\niteration {iterations}")
        start = time.time()

        rows, cols, channels = frame.shape

        # Prepare input data.
        if input_details[0]['dtype'] != np.uint8:
            input_data = cv2.dnn.blobFromImage(image=frame, scalefactor=1/255.0, size=input_shape[1:3], swapRB=True, crop=True)
            input_data = np.transpose(input_data, (0, 2, 3, 1))
        else:
            input_data = cv2.dnn.blobFromImage(image=frame, scalefactor=1, size=input_shape[1:3], swapRB=True, crop=True)
            input_data = np.transpose(input_data, (0, 2, 3, 1)).astype(np.uint8)

        # Retrieve what input image looks like
        input_reconstructed = cv2.cvtColor(input_data[0], cv2.COLOR_RGB2BGR)

        # Run model on the input data.
        start2 = time.time()
        detections = detect(interpreter, input_data)
        end2 = time.time()

        # Draw detections
        for det in detections:
            print(f" - detected class {det['class_id']} ({det['class_name']}) with confidence {det['confidence']:.2f}")
            draw_detection(input_reconstructed, det)

        # Show the image with a rectangle surrounding the detected objects 
        # cv2.imshow('input', frame)
        cv2.imshow('detections', input_reconstructed)

        # Compute Inference time and FPS counter
        end = time.time()
        print(f"{end2-start2:.3f}s inference time")
        running_inf_time += end2-start2
        print(f"{running_inf_time/iterations:.3f}s avg inference time")
        fps = 1 / (end - start)
        print(f"{fps:.2f} fps")
        running_fps += fps
        print(f"{running_fps/iterations:.2f} avg fps")

        
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cam.release()
# Destroy all the windows
cv2.destroyAllWindows()