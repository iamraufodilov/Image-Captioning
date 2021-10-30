# load libraries
from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

#extract feature from each photo in directory
def extract_features(directory):
    
    model = VGG16()
    model = Model(inputs = model.inputs, outputs = model.layers[-2].output)
    print(model.summary())

    features = dict()
    for name in listdir(directory):
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = name.split(',')[0]
        features[image_id] = feature
        print("%s" % name)

    return features

directory = "G:/rauf/STEPBYSTEP/Data2/image_captioning/Flicker8k_Dataset"
features = extract_features(directory)
print("Extracted features: %d" % len(features))
dump(features, open('features.pkl', 'wb'))
print("Featues extracted successfully!")