import numpy as np
import time
import cv2
import os

# define a video capture object
cam = cv2.VideoCapture(0)

DARKNET_PATH = 'tensorflow/yolo-model'
# Read labels that are used on object
labels = open(os.path.join(DARKNET_PATH, "coco.names")).read().splitlines()

# Make random colors with a seed, such that they are the same next time
np.random.seed(0)
colors = np.random.randint(0, 255, size=(len(labels), 3)).tolist()

# Give the configuration and weight files for the model and load the network.
net = cv2.dnn.readNetFromDarknet(os.path.join(DARKNET_PATH, "yolov3.cfg"),
                                 os.path.join(DARKNET_PATH, "yolov3.weights"))

# Determine the output layer, now this piece is not intuitive
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]



while(True):
    # Capture the video frame by frame
    success, frame = cam.read()
    if not success:
        print("Camera could not be read")
        break
    else:

        # Get the shape
        h, w = frame.shape[:2]
        # Load it as a blob and feed it to the network
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        # Get the output
        layer_outputs = net.forward(ln)
        end = time.time()

        # Initialize the lists we need to interpret the results
        boxes = []
        confidences = []
        class_ids = []
        
        # Loop over the layers
        for output in layer_outputs:
            
            # For the layer loop over all detections
            for detection in output:
                
                # The detection first 4 entries contains the object position and size
                scores = detection[5:]
                # Then it has detection scores - it takes the one with maximal score
                class_id = np.argmax(scores).item()
                # The maximal score is the confidence
                confidence = scores[class_id].item()
                # Ensure we have some reasonable confidence, else ignorre
                if confidence > 0.3:
                    # The first four entries have the location and size (center, size)
                    # It needs to be scaled up as the result is given in relative size (0.0 to 1.0)
                    box = detection[0:4] * np.array([w, h, w, h])
                    center_x, center_y, width, height = box.astype(int).tolist()
                    # Calculate the upper corner
                    x = center_x - width//2
                    y = center_y - height//2
                    # Add our findings to the lists
                    boxes.append([x, y, width, height])
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    
        # Only keep the best boxes of the overlapping ones
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)
        
        # Ensure at least one detection exists - needed otherwise flatten will fail
        if len(idxs) > 0:
            # Loop over the indexes we are keeping
            for i in idxs.flatten():
                # Get the box information
                x, y, w, h = boxes[i]
                # Make a rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), colors[class_ids[i]], 2)
                # Make and add text
                text = "{}: {:.4f}".format(labels[class_ids[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, colors[class_ids[i]], 2)
                
        # Write the image with boxes and text
        cv2.imshow("example.png", frame)



    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cam.release()
# Destroy all the windows
cv2.destroyAllWindows()