# Language Identification

Language identification of similar languages from text data using an ensemble of recurrent neural networks. Implementation based on the paper ["LIDE: Language Identification from Text
Documents"](https://arxiv.org/pdf/1701.03682.pdf).

## Install required packages

### Using conda

```bash
conda env create -f environment.yml
conda activate language_identification
```

### Using pip

```bash
pip install -r requirements.txt
```

## Running the notebook

After activating the environment, run the following command to start the notebook:

```bash
jupyter-notebook .
```

The entire project is contained in the Language_Ientification.ipynb file. You will always want to run the `Imports`, `Data`, `Preprocessing` and `Model` sections. If it your first time running the notebook, you will want to run `Make workspace` and `Download dataset` sections after the imports. `Hyperparameter search` and `Ensemble training` sections can be run independently, while `Testing` and `Inference` require the trained models to be saved on disk.

## Running the gradio frontend

After training or downloading the models, you can test the final model using the provided frontend. This requires the `gradio` package which is not included in the environment. Run the frontend using the following command:

```bash
python app.py
```

This will start a local server which you can access at http://localhost:7860 by default.

## Trained models

Trained models can be found from [here](https://www.dropbox.com/sh/k2aodc7bpe96zlh/AADNLmTiTTTGKbBXHEhGkHYaa?dl=0). Download all of them and unzip them in the `models` directory.

## Dataset

This project uses the [DSLCC v2.0](https://github.com/alvations/bayesmax/tree/master/bayesmax/data/DSLCC-v2.0) dataset from the [DSL Shared Task 2015](http://ttg.uni-saarland.de/lt4vardial2015/dsl.html). The corpus contains 20,000 instances per language (18,000 training + 2,000 development). Each instance is an excerpt extracted from journalistic texts and tagged with the country of origin of the text. A list of languages and the corresponding codes is shown in the following table:

<table>
    <tr>
        <th>Group Name</th>
        <th>Language Name</th>
        <th>Language Code</th>
    </tr>
    <tr>
        <td rowspan=2>South Eastern Slavic</td>
        <td>Bulgarian</td>
        <td>bg</td>
    </tr>
    <tr>
        <td>Macedonian</td>
        <td>mk</td>
    </tr>
    <tr>
        <td rowspan=3>South Western Slavic</td>
        <td>Bosnian</td>
        <td>bs</td>
    </tr>
    <tr>
        <td>Croatian</td>
        <td>hr</td>
    </tr>
    <tr>
        <td>Serbian</td>
        <td>sr</td>
    </tr>
    <tr>
        <td rowspan=2>West-Slavic</td>
        <td>Czech</td>
        <td>cz</td>
    </tr>
    <tr>
        <td>Slovak</td>
        <td>sk</td>
    </tr>
    <tr>
        <td rowspan=2>Ibero-Romance (Spanish)</td>
        <td>Peninsular Spanish</td>
        <td>es-ES</td>
    </tr>
    <tr>
        <td>Argentinian Spanish</td>
        <td>es-AR</td>
    </tr>
    <tr>
        <td rowspan=2>Ibero-Romance (Portugese)</td>
        <td>Brazilian Portugese</td>
        <td>pt-BR</td>
    </tr>
    <tr>
        <td>European Portugese</td>
        <td>pt-PT</td>
    </tr>
    <tr>
        <td rowspan=2>Astronesian</td>
        <td>Indonesian</td>
        <td>id</td>
    </tr>
    <tr>
        <td>Malay</td>
        <td>my</td>
    </tr>
    <tr>
        <td>Other</td>
        <td>Various Languages</td>
        <td>xx</td>
    </tr>
</table>

## Related links

- https://arxiv.org/pdf/1701.03682.pdf
- https://cs229.stanford.edu/proj2015/324_report.pdf
- https://cs229.stanford.edu/proj2015/324_poster.pdf
- https://sites.google.com/view/vardial2021/home
- http://ttg.uni-saarland.de/resources/DSLCC/
- https://mzampieri.com/publications.html
- https://mzampieri.com/papers/dsl2016.pdf

## Authors

- [Vladimir Vuksanović](https://github.com/VladimirV99)
- [Aleksa Kojadinović](https://github.com/aleksakojadinovic)
