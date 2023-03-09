import numpy as np
import cv2

# Get webcam video stream
video_stream = cv2.VideoCapture(0)

# Initialize confidence
confidence = 0.20

while True:
  # Get the current frame from video stream
  ret,current_frame = video_stream.read()
  # Use the video current frame instead of image
  img_to_detect = current_frame
  img_height = img_to_detect.shape[0]
  img_width = img_to_detect.shape[1]
  # Conversion to blob inorder to pass to model
  img_blob = cv2.dnn.blobFromImage(img_to_detect,0.003922,(416,416),swapRB=True,crop=False)
  # List of class labels
  class_labels = ['person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat',
                  'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
                  'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella',
                  'handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite',
                  'baseball bat','baseball glove','skateboard','surfboard','tennis racket',
                  'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich',
                  'orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa','pottedplant',
                  'bed','diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard',
                  'cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase',
                  'scissors','teddy bear','hair drier','toothbrush']
  # List of colors as array
  class_colors = ["0,255,0","0,0,255","255,0,0","255,255,0","0,255,255"]
  # Split list of colors based on ',' and change type to int
  class_colors = [np.array(every_color.split(",")).astype("int") for every_color in class_colors]
  # Convert list of colors to array
  class_colors = np.array(class_colors)
  # Apply color mask to the image numpy array
  class_colors = np.tile(class_colors,(16,1))

  # Loading pretrained model
  yolo_model = cv2.dnn.readNetFromDarknet('yolov4-tiny.cfg',
                                          'yolov4-tiny.weights')
  # Get all layers from the yolo network
  # Loop through the layers and obtain output layer
  yolo_layers = yolo_model.getLayerNames()
  yolo_output_layer = [yolo_layers[yolo_layer-1] for yolo_layer in yolo_model.getUnconnectedOutLayers()]

  # Input preprocessed blob into model
  yolo_model.setInput(img_blob)
  # Obtain detection layers by forwarding via output layer
  obj_detection_layers = yolo_model.forward(yolo_output_layer)

  # Loop over each of the layer outputs
  for object_detection_layer in obj_detection_layers:
    # Loop over the detections
    for object_detection in object_detection_layer:
      # object_detection[1 to 4] >> will have center points,box width and box height
      # object_detection[5] >> will have scores for all objects within bounding box
      all_scores = object_detection[5:]
      predicted_class_id = np.argmax(all_scores)
      prediction_confidence = all_scores[predicted_class_id]

      # Take predictions with confidence greater than set prediction confidence
      if prediction_confidence > confidence:
        # Get the predicted label
        predicted_class_label = class_labels[predicted_class_id]
        # Obtain the bounding box coordinates for actual image from resized image size
        bounding_box = object_detection[0:4]*np.array([img_width,img_height,img_width,img_height])
        (box_center_x_pt,box_center_y_pt,box_width,box_height) = bounding_box.astype('int')
        start_x_pt = int(box_center_x_pt-(box_width/2))
        start_y_pt = int(box_center_y_pt-(box_height/2))
        end_x_pt = start_x_pt + box_width
        end_y_pt = start_y_pt + box_height

        # Get a random mask color from the numpy array of colors
        box_color = class_colors[predicted_class_id]
        # Convert the color numpy array as a list and apply to text and box
        box_color = [int(c) for c in box_color]
        # Print the prediction in console
        predicted_class_label = "{}: {:.2f}%".format(predicted_class_label,prediction_confidence*100)
        print("Predicted Object {}".format(predicted_class_label))

        # Draw rectangle and text in the image
        cv2.rectangle(img_to_detect,(start_x_pt,start_y_pt),(end_x_pt,end_y_pt),box_color,1)
        cv2.putText(img_to_detect,predicted_class_label,(start_x_pt,start_y_pt-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,box_color,1)
  
  cv2.imshow(img_to_detect)
  # Terminate while loop if 'q' key is pressed
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Releasing the stream
# Close all opencv windows
video_stream.release()
cv2.destroyAllWindows()


  