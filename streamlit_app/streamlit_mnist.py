# adapted from https://github.com/rahulsrma26/streamlit-mnist-drawable/blob/dev/app.py
import os
import numpy as np
from bentoml.client import Client
from PIL import Image

import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas


client = Client.from_url("http://localhost:3000")
#model = load_model('model')
# st.markdown('<style>body{color: White; background-color: DarkSlateGrey}</style>', unsafe_allow_html=True)

st.title('My Digit Recognizer')
st.markdown('''
Try to write a digit!
''')

# data = np.random.rand(28,28)
# img = cv2.resize(data, (256, 256), interpolation=cv2.INTER_NEAREST)

SIZE = 192
mode = st.checkbox("Draw (or Delete)?", True)
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw" if mode else "transform",
    key='canvas')

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('Model Input')
    st.image(rescaled)

# curl as python call
#curl -H "Content-Type: multipart/form-data" -F'fileobj=@samples/0.png;type=image/png' http://127.0.0.1:3000/predict_image
def call_api():
    with open('output.jpg', 'rb') as f:
        file = Image.open(f)
        resp = client.call('predict_image', file)
        return resp
    
if st.button('Predict'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(test_x)
    img = img.convert("RGB")
    img.save('output.jpg')
    res = call_api()
    #val = model.predict(test_x.reshape(1, 28, 28))
    #val = client.call("predict_image", img)
    st.write(f'result: {np.argmax(res[0])}')
    st.bar_chart(res[0])

# docker build -t g4challenge/bentoml_tensorflow_mnist:latest .