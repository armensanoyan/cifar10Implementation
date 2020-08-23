import tensorflow as tf
from tensorflow.keras import layers, models,datasets
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (tast_images, tast_labels) = datasets.cifar100.load_data()

train_images, tast_images = train_images/255.0, tast_images/255.0

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if( logs.get('accuracy') > 0.95):
            print('The accuracy is higher then 95%')
            self.model.stop_training = True

callbacks = myCallback()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics='accuracy'
)

history = model.fit(
    train_images,
    train_labels,
    epochs=25,
    validation_data=(tast_images, tast_labels),
    callbacks=[callbacks]
)
# showes accuracy depended on amount of epochs
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

#measures the accuracy
test_loss, test_acc = model.evaluate(tast_images,  tast_labels, verbose=2)

# check if it works as expacted
model.save('../models/cifar100.h5') 



















# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     # The CIFAR labels happen to be arrays, 
#     # which is why you need the extra index
#     plt.xlabel(class_names[train_labels[i][0]])
# plt.show()