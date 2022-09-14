from itertools import chain
import pickle
from nltk.util import ngrams
import numpy as np
import gradio as gr
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json, text_to_word_sequence
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

def sentence_to_char_ngram(sentence, n):
    s = ''.join([c if c not in '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“„”–' else ' ' for c in sentence])
    tokens = text_to_word_sequence(s)
    ngrams_ = [[''.join(ng) for ng in list(ngrams(token, n))] for token in tokens if len(token) >= n]
    return ' '.join(chain.from_iterable(ngrams_))

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
    
    input_c2 = sentence_to_char_ngram(input_sentence, 2)
    input_c2 = tokenizer_c2.texts_to_sequences([input_c2])
    input_c2 = pad_sequences(input_c2, padding='post', truncating='post', maxlen=150)
    input_c2 = to_categorical(input_c2, num_classes=(len(tokenizer_c2.word_index.keys()) + 1))
    prediction_c2 = model_c2.predict(input_c2)
    
    input_c3 = sentence_to_char_ngram(input_sentence, 3)
    input_c3 = tokenizer_c3.texts_to_sequences([input_c3])
    input_c3 = pad_sequences(input_c3, padding='post', truncating='post', maxlen=150)
    input_c3 = to_categorical(input_c3, num_classes=tokenizer_c3.num_words)
    prediction_c3 = model_c3.predict(input_c3)
    
    prediction_ensemble = ensemble.predict_proba(
        np.hstack((prediction_w1, prediction_c2, prediction_c3))
    )[0]
    
    return dict(zip(map(lambda code: CLASS_MAP[code], ensemble.classes_), prediction_ensemble))

if __name__ == '__main__':
    # Load models
    model_w1, tokenizer_w1 = load_rnn('model_w1_final')
    model_c2, tokenizer_c2 = load_rnn('c2_model')
    model_c3, tokenizer_c3 = load_rnn('c3_model')
    
    with open('models/ensemble_final/model.pkl', 'rb') as f:
        ensemble = pickle.load(f)
    
    # Create interface
    iface = gr.Interface(
        predict,
        gr.inputs.Textbox(lines=5, placeholder='Enter your text here', label='input'),
        'label',
        # TODO add some examples
        examples=[
            ['Mašinsko učenje je oblast veštačke inteligencije koja se bavi izgradnjom računarskih sistema koji uče iz iskustva. Ova oblast je u poslednjih nekoliko godina izuzetno popularna, kako u akademskim krugovima, tako i u industriji.'],
            ['Calangulango, do calango da pretinha, \'Tô cantando essa mudinha pra senhora se lembrar, Daquele tempo que vivia la na roça com uma filha Na barriga e outra filha pra criar'],
            ['Vykazuje též poměrně nízkou nerovnost mezi nejbohatšími a nejchudšími obyvateli a relativně vyvážené přerozdělování bohatství napříč populací. Míra nezaměstnanosti je dlouhodobě nízká a pod průměrem'],
            ['Secara lazimnya, rumpun bahasa Austronesia dibahagi kepada beberapa kelompok. Dua kelompok utama ialah bahasa Taiwan dan bahasa Melayu-Polinesia. Kemudian rumpun bahasa Melayu-Polinesia dibahagi pula menjadi bahasa-bahasa Melayu-Polinesia Barat, Tengah dan Timur.']
        ],
        title='Language Identification',
        description='Language identification of similar languages using an ensemble of recurrent neural networks. Implementation based on the paper ["LIDE: Language Identification from Text Documents"](https://arxiv.org/pdf/1701.03682.pdf).',
        allow_flagging='never'
    )

    iface.launch()
