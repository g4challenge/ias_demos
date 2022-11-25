#pip install bentoml scikit-learn pandas orange3
#%%
import numpy as np
import PIL.Image

import bentoml

runner = bentoml.keras.get("tensorflow_mnist:latest").to_runner()
runner.init_local()  # for debug only. please do not call this in the service

img = PIL.Image.open("samples/0.png")
arr = np.array(img) / 255.0
arr = arr.astype("float32")

# add color channel dimension for greyscale image
arr = np.expand_dims(arr, 2)


runner.run(list(arr))

#runner.run(arr)  # => returns an array of probabilities for numbers 0-9

#%%
#model =bentoml.keras.load_model("tensorflow_mnist:latest")

# bentofile.yaml
# bentoml build

# bentoml containerize tensorflow_mnist_demo:latest