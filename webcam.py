
# import the opencv library
import cv2
import tensorflow as tf
import numpy as np
model = tf.keras.models.load_model('final_model.h5')
  
# define a video capture object
vid = cv2.VideoCapture("p002_c2.avi")
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
  
    # Display the resulting frame
    #cv2.imshow('frame', frame)
    #img = cv2.imread(frame)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # this step is import turning BGR to RGB
    img = cv2.resize(img,(64,64))

    img = np.array(img).reshape(64,64,3)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    model.predict_classes(img)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    cv2.imshow('capture', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
