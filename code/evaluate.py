import numpy as np
import tensorflow as tf
# from PIL import Image
from tensorflow.keras.preprocessing import image

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck']

model = tf.keras.models.load_model('/Users/armensanoyan/simpleAlgoritms/clothesGuesser/convolution/allkindofstuff/rps.h5')
img = image.load_img("/Users/armensanoyan/simpleAlgoritms/clothesGuesser/convolution/allkindofstuff/data/horse.jpg", target_size=(32, 32))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)

predictions = np.argmax(classes)
print('\n',"It's",class_names[predictions])
# print('predictions', predictions)
