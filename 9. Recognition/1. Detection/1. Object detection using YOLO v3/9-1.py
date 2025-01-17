'''
coco_names.txt: Name of categories in COCO dataset
yolov3.weights: Weights of neural networks
yolov3.cfg: Structure of the neural network
'''
import numpy as np
import cv2 as cv
import sys

def construct_yolo_v3():    # Construct YOLO model
    f = open('coco_names.txt', 'r')
    class_names = [line.strip() for line in f.readlines()]  # class(category) names
    
    model = cv.dnn.readNet('yolov3.weights', 'yolov3.cfg')  # Construct YOLO model object
    layer_names = model.getLayerNames()
    out_layers = [layer_names[i-1] for i in model.getUnconnectedOutLayers()]    # get YOLO_82(big), 94(mid), 106(small) floor
    
    return model, out_layers, class_names

def yolo_detect(img, yolo_model, out_layers):   # Use YOLO to detect objects in the image
    height, width = img.shape[0], img.shape[1]  # Save img's original info
    test_img = cv.dnn.blobFromImage(img, 1.0/256,(448,448),(0,0,0), swapRB=True)    
    # blobFromImage function: transform img to an applicable form for YOLO input
    # change pixel info that is [0,255] into [0,1], reshape size to 488 X 488, change BGR to RGB
    yolo_model.setInput(test_img)   # insert modified img in the model
    output3 = yolo_model.forward(out_layers)    # foward propagation and save the three tensors of each floor
    
    box, conf, id = [],[],[]    # Box, Confidence score, Category number
    for output in output3:  # The three floors are for diverse box sizes
        for vec85 in output:    
            # Deal with the 85d vector ((x, y): center of grid, (w, h): anchor box, confidence: o, class prob: p1, p2, ..., p80)
            # Each vec85 is an 85-dimensional vector corresponding to one anchor box in a specific grid cell
            scores = vec85[5:]  # get all the class probs
            class_id = np.argmax(scores)    # get the class with the highest prob
            confidence = scores[class_id]   # get the confidence score of the highest class
            if confidence > 0.5:
                # change the box info back to the original coordinate system
                centerx, centery = int(vec85[0]*width), int(vec85[1]*height)
                w, h = int(vec85[2]*width), int(vec85[3]*height)
                x, y = int(centerx-w/2), int(centery-h/2)
                # save the box, confidence score, and class name
                box.append([x, y, x+w, y+h])
                conf.append(float(confidence))
                id.append(class_id)
    # If we finish here, there are lots of overlapping boxes
            
    ind = cv.dnn.NMSBoxes(box, conf, 0.5, 0.4)  # Apply Non-Maximum Suppresion to remove duplicates
    objects = [box[i] + [conf[i]] + [id[i]] for i in range(len(box)) if i in ind]   # save the three info in one object
    return objects

# Main
model, out_layers, class_names = construct_yolo_v3()
colors = np.random.uniform(0, 255, size=(len(class_names),3))

img = cv.imread("bus_1713.png")

res = yolo_detect(img, model, out_layers)

for i in range(len(res)):
    x1, y1, x2, y2, confidence, id = res[i]
    text = str(class_names[id] + '%.3f'%confidence)
    cv.rectangle(img, (x1, y1), (x2, y2), colors[id], 2)
    cv.putText(img, text, (x1, y1+30), cv.FONT_HERSHEY_PLAIN, 1.5, colors[id], 2)
    
cv.imshow("Object detection by YOLO v.3", img)

cv.waitKey()
cv.destroyAllWindows()
