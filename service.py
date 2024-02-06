# service.py
import typing as t

import numpy as np
from PIL.Image import Image as PILImage

import bentoml
from bentoml.io import Image
from bentoml.io import NumpyNdarray
import logging

ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

bentoml_logger = logging.getLogger("bentoml")
bentoml_logger.addHandler(ch)
bentoml_logger.setLevel(logging.DEBUG)

mnist_runner = bentoml.keras.get("tensorflow_mnist").to_runner()

svc = bentoml.Service(
    name="tensorflow_mnist_demo",
    runners=[
        mnist_runner,
    ],
)


@svc.api(input=Image(), output=NumpyNdarray(dtype="int64"))
async def predict_image(f: PILImage) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
    assert isinstance(f, PILImage)
    # resize
    f = f.resize((28, 28))
    arr = np.array(f)/255.0
    if arr.shape != (28, 28):
        # transform to greyscale
        arr = np.dot(arr[..., :3], [0.299, 0.587, 0.114])
        arr = arr.astype("float32")
    #bentoml_logger.info(f"arr: {arr}")
    bentoml_logger.info(f"arr.shape: {arr.shape}")
    assert arr.shape == (28, 28)

    # extra channel dimension
    arr = np.expand_dims(arr, (0, 3)).astype("float32")
    
    output_tensor = await mnist_runner.async_run(arr)
    return output_tensor