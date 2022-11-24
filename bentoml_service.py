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



#from bentoml.adapters import DataframeInput
#from bentoml.frameworks.sklearn import SklearnModelArtifact

#%%
# Load Model from Pickle
import pickle
with open("save_rf.pkcls", "rb") as f:
    clf = pickle.load(f)

# Save model to the BentoML local model store
saved_model = bentoml.sklearn.save_model("iris_clf", clf)
print(f"Model saved: {saved_model}")


# Model saved: Model(tag="iris_clf:zy3dfgxzqkjrlgxi")
#%% writefile bentoml.py
import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
     result = iris_clf_runner.predict.run(input_series)
     return result



@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([SklearnModelArtifact('model')])
class IrisClassifier(bentoml.BentoService):

    @bentoml.api(input=DataframeInput(), batch=True)
    def predict(self, df):
        return self.artifacts.model.predict(df)