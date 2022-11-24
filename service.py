# service.py
import typing as t

import numpy as np
import PIL.Image
from PIL.Image import Image as PILImage

import bentoml
from bentoml.io import Image
from bentoml.io import NumpyNdarray


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
    arr = np.array(f)/255.0
    assert arr.shape == (28, 28)

    # extra channel dimension
    arr = np.expand_dims(arr, (0, 3)).astype("float32")
    
    output_tensor = await mnist_runner.async_run(arr)
    return output_tensor