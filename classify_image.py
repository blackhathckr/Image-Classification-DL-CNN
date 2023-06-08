import cv2 
import joblib as jb 

IMAGE_WIDTH=128
IMAGE_HEIGHT=128

model=jb.load('classifier_model.h5')

img = cv2.imread('testing_image_path')
img = cv2.resize(img,(IMAGE_HEIGHT,IMAGE_WIDTH))
original = img.copy()
img = img.reshape(1,128,128,3)
prediction = model.predict(img)
print(['label_1','label_2'][int(prediction[0][0])])
cv2.imshow('Live predictions',original)
cv2.waitKey(0)
cv2.destroyAllWindows()