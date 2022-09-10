import pickle
import numpy as np
import gradio as gr
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.utils import to_categorical

CLASS_MAP = {
    'bg': 'Bulgarian',
    'mk': 'Macedonian',
    'bs': 'Bosnian',
    'hr': 'Croatian',
    'sr': 'Serbian',
    'cz': 'Czech',
    'sk': 'Slovak',
    'es-ES': 'Peninsular Spanish',
    'es-AR': 'Argentinian Spanish',
    'pt-BR': 'Brazilian Portuguese',
    'pt-PT': 'European Portuguese',
    'id': 'Indonesian',
    'my': 'Malay',
    'xx': 'Other'
}

def load_rnn(model_name):
    model = load_model(f'models/{model_name}')
    with open(f'models/{model_name}/tokenizer.json', 'r') as f:
        tokenizer = tokenizer_from_json(f.read())
    return model, tokenizer

def predict(input_sentence):
    input_w1 = tokenizer_w1.texts_to_sequences([input_sentence])
    input_w1 = pad_sequences(input_w1, 50, padding='post', truncating='post')
    input_w1 = to_categorical(input_w1, num_classes=tokenizer_w1.num_words)
    prediction_w1 = model_w1.predict(input_w1)
    
    prediction_ensemble = ensemble.predict_proba(
        # TODO remove reshape after adding more models to hstack
        np.hstack((prediction_w1)).reshape(1,-1)
    )[0]
    
    return dict(zip(map(lambda code: CLASS_MAP[code], ensemble.classes_), prediction_ensemble))

if __name__ == '__main__':
    # Load models
    model_w1, tokenizer_w1 = load_rnn('model_w1_final')
    
    with open('models/ensemble_final/model.pkl', 'rb') as f:
        ensemble = pickle.load(f)
    
    # Create interface
    iface = gr.Interface(
        predict,
        gr.inputs.Textbox(lines=5, placeholder='Enter your text here', label='input'),
        'label',
        # TODO add some examples
        examples=[
        ],
        title='Language Identification',
        description='Language identification of similar languages using an ensemble of recurrent neural networks. Implementation based on the paper ["LIDE: Language Identification from Text Documents"](https://arxiv.org/pdf/1701.03682.pdf).',
        allow_flagging='never'
    )

    iface.launch()
