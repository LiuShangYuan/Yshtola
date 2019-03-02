# Yshtola

## Requirements
* python 3.5+
* tensorflow 1.13+
* scipy

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

