import os, glob, kagglehub
import tensorflow as tf
from tensorflow.keras import layers

# Dataset download (benign vs malignant)
path = kagglehub.dataset_download("fanconic/skin-cancer-malignant-vs-benign")
train_dir = os.path.dirname(glob.glob(path + "/**/train/benign", recursive=True)[0])
test_dir = train_dir.replace("train", "test")
print(os.listdir(train_dir))

IMG = (224, 224)
train = tf.keras.utils.image_dataset_from_directory(
    train_dir, image_size=IMG, batch_size=32,
    label_mode="binary", validation_split=0.2, subset="training", seed=1)
val = tf.keras.utils.image_dataset_from_directory(
    train_dir, image_size=IMG, batch_size=32,
    label_mode="binary", validation_split=0.2, subset="validation", seed=1)
print(train.class_names)   # ['benign', 'malignant'] aana chahiye

base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base.trainable = False

inp = layers.Input((224, 224, 3))          # app 0-1 wali image bhejti hai
x = layers.Rescaling(255.0)(inp)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inp, out)

# images ko 0-1 mein karo (app bhi yehi karti hai)
norm = layers.Rescaling(1/255.0)
train = train.map(lambda a, b: (norm(a), b))
val = val.map(lambda a, b: (norm(a), b))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train, validation_data=val, epochs=8)

model.save("skin_cancer_model.keras")
from google.colab import files
files.download("skin_cancer_model.keras")
