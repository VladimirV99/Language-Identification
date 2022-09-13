import os

import requests

os.makedirs(os.path.join('datasets/DSLCC-v2.0'), exist_ok=True)
if not os.path.exists('models'):
    os.mkdir('models')


train_url = 'https://raw.githubusercontent.com/alvations/bayesmax/master/bayesmax/data/DSLCC-v2.0/train-dev/train.txt'
devel_url = 'https://raw.githubusercontent.com/alvations/bayesmax/master/bayesmax/data/DSLCC-v2.0/train-dev/devel.txt'
test_url = 'https://raw.githubusercontent.com/alvations/bayesmax/master/bayesmax/data/DSLCC-v2.0/test/test.txt'

if not os.path.exists('datasets/DSLCC-v2.0/train.txt'):
    print('Downloading train set...')
    train_content = requests.get(train_url).content
    open("datasets/DSLCC-v2.0/train.txt", "wb").write(train_content)

if not os.path.exists('datasets/DSLCC-v2.0/devel.txt'):    
    print('Downloading devel set...')
    devel_content = requests.get(devel_url).content
    open("datasets/DSLCC-v2.0/devel.txt", "wb").write(devel_content)


if not os.path.exists('datasets/DSLCC-v2.0/test.txt'):
    print('Downloading test set...')
    test_content = requests.get(test_url).content
    open("datasets/DSLCC-v2.0/test.txt", "wb").write(test_content)





