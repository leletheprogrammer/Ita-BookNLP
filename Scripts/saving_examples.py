import os
from sklearn.model_selection import train_test_split
from spacy.tokens import DocBin
from spacy.cli.init_config import init_config_cli, Optimizations
from pathlib import Path
from thinc.api import Config

def splitting_set(documents: list):
    training_set, validation_set = train_test_split(documents, test_size = 0.2)
    validation_set, test_set = train_test_split(validation_set, test_size = 0.5)
    
    return training_set, validation_set, test_set

def saving_set(elements_set: list, file: str):
    binary_documents = DocBin()
    for document in elements_set:
        binary_documents.add(document)
    binary_documents.to_disk(file)

def split_save(documents: list, output_directory: str):
    training_set, validation_set, test_set = splitting_set(documents)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    saving_set(training_set, os.path.join(output_directory, 'training.spacy'))
    saving_set(validation_set, os.path.join(output_directory, 'dev.spacy'))
    saving_set(test_set, os.path.join(output_directory, 'test.spacy'))

def initialize_config_file(output_directory: str, batch_size: int, dropout: float, patience: int, max_epochs: int, max_steps: int, eps: float, learn_rate: float):
    init_config_cli(output_file = Path(output_directory + '/config.cfg'), lang = 'it', pipeline = 'ner', optimize = Optimizations.efficiency, gpu = False, pretraining = False)
    
    base_config = Config().from_disk(Path(output_directory + '/config.cfg'))
    update_config = Config({'nlp': {'batch_size': batch_size}, 'training': {'dropout': dropout, 'patience': patience, 'max_epochs': max_epochs, 'max_steps': max_steps, 'optimizer': {'eps': eps, 'learn_rate': learn_rate}}})
    Config(base_config).merge(update_config).to_disk(Path(output_directory + '/config.cfg'))