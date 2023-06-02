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

MODEL_PATH = "./models/model1-train4/best_int8.tflite"


while(True):
    # Capture the video frame by frame
    success, frame = cam.read()

    if not success:
        print("Camera could not be read")
        break

    else:
        rows, cols, channels = frame.shape

        print(frame.shape)
 
        ## Load TFLite model and allocate tensors.
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Prepare input data.
        input_shape = input_details[0]['shape']
        input_data = cv2.dnn.blobFromImage(image=frame, scalefactor=1/255.0, size=(640, 640), swapRB=True, crop=True)
        input_data = np.transpose(input_data, (0, 2, 3, 1))
        print(input_data.shape)

        #TODO: check what input image looks like
        input_reconstructed = cv2.cvtColor(input_data[0], cv2.COLOR_RGB2BGR)


        # Run model on the input data.
        interpreter.set_tensor(input_details[0]['index'], input_data)
        start = time.time()
        interpreter.invoke()
        end = time.time()
        fps = 1 / (end - start)
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # print(output_data)
        print(fps, "fps")
        print(len(output_data[0]), "detections")

        #TODO: Check that this is how to analyse outputs

        # Loop on the outputs
        for detection in output_data[0]:
            score = float(detection[2])
            if score >= 0:
                print("pass")
                left = detection[3] #* cols
                top = detection[4] #* rows
                right = detection[5] #* cols
                bottom = detection[6] #* rows
                # print(left, top, right, bottom)
        
                #draw a red rectangle around detected objects
                cv2.rectangle(input_reconstructed, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
                cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
                
                # TODO: Add class labels and confidence value on image

        #TODO: Add FPS counter

        # Show the image with a rectangle surrounding the detected objects 
        # cv2.imshow('detections', frame)
        cv2.imshow('input_reconstructed', input_reconstructed)

        

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cam.release()
# Destroy all the windows
cv2.destroyAllWindows()