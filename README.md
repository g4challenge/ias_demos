# Example Intelligent Adaptive Systems

In this Repository, I collect Orange3 examples and python scripts for Intelligent Adaptive Systems.

The python code can be executed in cells like in Jupyter Notebooks. The Orange3 examples can be executed in Orange3.

## Orange3 Examples


## Code Examples

`classification.py` shows a simple classification example in interactive Python.

`regression.py` shows a simple regression example in interactive Python.

`clustering.py` shows a simple clustering example in interactive Python.

`train_metal.py` shows a simple example classifying digits in interactive Python on the MNIST dataset, using a neural network and acceleration on the GPU (on Macs with Metal).

`bentoml_service.py` shows a simple example classifying digits in interactive Python on the MNIST dataset, using a neural network and acceleration on the GPU (on Macs with Metal), and exporting the model as a BentoML service, which then can be used by a REST API and a streamlit web app.

`streamlit_app/streamlit_mnist.py` shows a streamlit web app for the MNIST classification example, accessing the BentoML service.