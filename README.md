# Yshtola

## Requirements
* python 3.5+
* tensorflow 1.13+
* scipy
* tdqm
* nltk

## Usage

### Download the dataset
    $ cd data/
    $ chmod +x ./data_download.sh
    $ sudo bash ./data_download.sh

### Preprocess data
    $ python prepro.py
    
### Pretrain model
    $ python main.py --mode='pretrain'
    
### Train model
    $ python main.py --mode='train'

## Reference Data Resources
* Amazon reviews dataset: [http://jmcauley.ucsd.edu/data/amazon/](link)
* MPQA Subjectivity Lexicon: [https://mpqa.cs.pitt.edu/lexicons/subj_lexicon/](Link)
