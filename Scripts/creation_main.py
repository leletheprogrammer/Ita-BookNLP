import os, sys
from spacy import load, blank
from preparation import creation_set, preparation_training
from training_evaluation import training_models, evaluate_models
from pathlib import Path

def creation_models(preparation_set: list, output_directory: str, model: str, hprpms_fir_lev: dict, hprpms_sec_lev: dict, hprpms_thi_lev: dict, hprpms_fou_lev: dict):
    print('Starting preprocessing')
    preparation_training(preparation_set, output_directory, model, hprpms_fir_lev, hprpms_sec_lev, hprpms_thi_lev, hprpms_fou_lev)
    print('Preprocessing finished')
    
    print('Starting training')
    training_models(output_directory)
    print('Training finished')
    
    print('Starting evaluation')
    evaluate_models(output_directory)
    print('Evaluation finished')

def create_save_model(output_directory: str):
    first_level_nlp = load(os.path.join(output_directory, 'first_level', 'models', 'model-best'))
    first_level_nlp.replace_listeners('tok2vec', 'ner', ['model.tok2vec'])
    
    second_level_nlp = load(os.path.join(output_directory, 'second_level', 'models', 'model-best'))
    second_level_nlp.replace_listeners('tok2vec', 'ner', ['model.tok2vec'])
    
    third_level_nlp = load(os.path.join(output_directory, 'third_level', 'models', 'model-best'))
    third_level_nlp.replace_listeners('tok2vec', 'ner', ['model.tok2vec'])
    
    fourth_level_nlp = load(os.path.join(output_directory, 'fourth_level', 'models', 'model-best'))
    fourth_level_nlp.replace_listeners('tok2vec', 'ner', ['model.tok2vec'])
    
    new_nlp = blank('it')
    new_nlp.add_pipe(factory_name = 'ner', name = 'first_level_nlp', source = first_level_nlp, first = True)
    new_nlp.add_pipe(factory_name = 'ner', name = 'second_level_nlp', source = second_level_nlp, last = True)
    new_nlp.add_pipe(factory_name = 'ner', name = 'third_level_nlp', source = third_level_nlp, last = True)
    new_nlp.add_pipe(factory_name = 'ner', name = 'fourth_level_nlp', source = fourth_level_nlp, last = True)
    new_nlp.to_disk(os.path.join(output_directory, 'new-model'))

if __name__ == '__main__':
    brats_directory = str(Path(sys.argv[1]))
    preparation_set = creation_set(brats_directory)

    output_directory = str(Path(sys.argv[2]))
    
    model = 'it_core_news_sm'
    hprpms_fir_lev = dict(batch_size = 1024, dropout = 0.1, patience = 1000, max_epochs = 21, max_steps = 10000, eps = 0.00000001, learn_rate = 0.001)
    hprpms_sec_lev = dict(batch_size = 1024, dropout = 0.1, patience = 1000, max_epochs = 21, max_steps = 10000, eps = 0.00000001, learn_rate = 0.001)
    hprpms_thi_lev = dict(batch_size = 1024, dropout = 0.2, patience = 1000, max_epochs = 21, max_steps = 10000, eps = 0.00000001, learn_rate = 0.001)
    hprpms_fou_lev = dict(batch_size = 1024, dropout = 0.2, patience = 1000, max_epochs = 21, max_steps = 10000, eps = 0.00000001, learn_rate = 0.001)
    creation_models(preparation_set, os.path.join(output_directory, 'small'), model, hprpms_fir_lev, hprpms_sec_lev, hprpms_thi_lev, hprpms_fou_lev)
    create_save_model(os.path.join(output_directory, 'small'))
    
    model = 'it_core_news_md'
    hprpms_fir_lev = dict(batch_size = 1024, dropout = 0.1, patience = 1000, max_epochs = 21, max_steps = 10000, eps = 0.00000001, learn_rate = 0.001)
    hprpms_sec_lev = dict(batch_size = 1024, dropout = 0.2, patience = 1000, max_epochs = 21, max_steps = 10000, eps = 0.00000001, learn_rate = 0.001)
    hprpms_thi_lev = dict(batch_size = 1024, dropout = 0.1, patience = 1000, max_epochs = 21, max_steps = 10000, eps = 0.00000001, learn_rate = 0.001)
    hprpms_fou_lev = dict(batch_size = 1024, dropout = 0.2, patience = 1000, max_epochs = 21, max_steps = 10000, eps = 0.00000001, learn_rate = 0.001)
    creation_models(preparation_set, os.path.join(output_directory, 'medium'), model, hprpms_fir_lev, hprpms_sec_lev, hprpms_thi_lev, hprpms_fou_lev)
    create_save_model(os.path.join(output_directory, 'medium'))
    
    model = 'it_core_news_lg'
    hprpms_fir_lev = dict(batch_size = 1024, dropout = 0.2, patience = 1000, max_epochs = 21, max_steps = 10000, eps = 0.00000001, learn_rate = 0.001)
    hprpms_sec_lev = dict(batch_size = 1024, dropout = 0.1, patience = 1000, max_epochs = 21, max_steps = 10000, eps = 0.00000001, learn_rate = 0.001)
    hprpms_thi_lev = dict(batch_size = 1024, dropout = 0.2, patience = 1000, max_epochs = 21, max_steps = 10000, eps = 0.00000001, learn_rate = 0.001)
    hprpms_fou_lev = dict(batch_size = 1024, dropout = 0.2, patience = 1000, max_epochs = 21, max_steps = 10000, eps = 0.00000001, learn_rate = 0.001)
    creation_models(preparation_set, os.path.join(output_directory, 'large'), model, hprpms_fir_lev, hprpms_sec_lev, hprpms_thi_lev, hprpms_fou_lev)
    create_save_model(os.path.join(output_directory, 'large'))