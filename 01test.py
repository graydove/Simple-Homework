import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

'''
Classifications:
0 T-shirt/top
1 Trouser
2 Pullover
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle boot
'''
classifications = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'}

img = Image.open("test03.bmp").convert('L')
img = img.resize((28, 28))
input_image = image.img_to_array(img)
input_image = np.expand_dims(input_image, axis=0)

model = load_model(r"fashion_model.h5")
y = model.predict_classes(input_image)[0]

plt.imshow(img)
plt.title("The Clothing Class of the Photo is: %s %s" % (y, classifications[y]))
plt.axis('off')
plt.show()
