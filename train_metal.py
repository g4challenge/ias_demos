# Tensorflow Setup for GPU
# conda create -n tf-metal
# conda activate tf-metal
# conda install -c apple tensorflow-deps
# python -m pip install tensorflow-macos
# python -m pip install tensorflow-metal
# python -m pip install tensorflow_datasets
# python -m pip install matplotlib

#%%

# Normal Setup for Users:
# pip install tensorflow
# pip install tensorflow_datasets

import tensorflow as tf
import tensorflow_datasets as tfds

BATCH_SIZE = 64
EPOCHS = 10
AUTOTUNE = tf.data.AUTOTUNE

# load MNIST
print('\ndownload mnist')
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
# %%
def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

#%%
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.cache()
ds_test = ds_test.batch(BATCH_SIZE)

print('\ncreate and compile model')
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

model.fit(ds_train, epochs=EPOCHS, validation_data=ds_test)
# %%
#from matplotlib import pyplot
#X_train = ds_train
#pyplot.imshow(X_train[0][0], cmap=pyplot.get_cmap('gray'))
# pyplot.show()


# %%
import bentoml
#bentoml.tensorflow.save_model("tensorflow_mnist", model)
bentoml.keras.save_model("tensorflow_mnist", model)
# bentoml models list
# bentoml list 
# %%
# bentoml serve service:svc --reload   
# bentoml build 
# bentoml containerize tensorflow_mnist_demo:latest


# %%
# Streamlit
# streamlit run streamlit_app/streamlit_mnist.py
# Docker Streamlit
# docker build -t streamlit_mnist_demo .
# docker run -p 8501:8501 streamlit_mnist_demo