import keras
import keras.layers as layers

(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

# Before
# model = keras.Sequential([
#     layers.Flatten(input_shape=[28, 28]),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(10, activation='softmax')
# ])
# After
model = keras.Sequential([
    layers.Flatten(input_shape=[28, 28]),
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Before
# model.fit(train_images, train_labels, epochs=10)
# After
model.fit(train_images, train_labels, epochs=150)
mp = "fashion_model.h5"
model.save(mp)

model.evaluate(test_images, test_labels)
