import os, sys
from spacy import load
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

def create_save_model(output_directory: str, model: str):
    first_level_nlp = load(os.path.join(output_directory, 'first_level', 'models', 'model-best'))
    first_level_nlp.replace_listeners('tok2vec', 'ner', ['model.tok2vec'])
    
    second_level_nlp = load(os.path.join(output_directory, 'second_level', 'models', 'model-best'))
    second_level_nlp.replace_listeners('tok2vec', 'ner', ['model.tok2vec'])
    
    third_level_nlp = load(os.path.join(output_directory, 'third_level', 'models', 'model-best'))
    third_level_nlp.replace_listeners('tok2vec', 'ner', ['model.tok2vec'])
    
    fourth_level_nlp = load(os.path.join(output_directory, 'fourth_level', 'models', 'model-best'))
    fourth_level_nlp.replace_listeners('tok2vec', 'ner', ['model.tok2vec'])
    
    new_nlp = load(model)
    new_nlp.add_pipe(factory_name = 'ner', name = 'first_level_nlp', source = first_level_nlp, last = True)
    new_nlp.add_pipe(factory_name = 'ner', name = 'second_level_nlp', source = second_level_nlp, last = True)
    new_nlp.add_pipe(factory_name = 'ner', name = 'third_level_nlp', source = third_level_nlp, last = True)
    new_nlp.add_pipe(factory_name = 'ner', name = 'fourth_level_nlp', source = fourth_level_nlp, last = True)
    new_nlp.to_disk(os.path.join(output_directory, 'new-model'))

def create_new_models(output_directory: str):
    create_save_model(os.path.join(output_directory, 'small'), 'it_core_news_sm')
    create_save_model(os.path.join(output_directory, 'medium'), 'it_core_news_md')
    create_save_model(os.path.join(output_directory, 'large'), 'it_core_news_lg')

def create_train_evaluation():
    brats_directory = str(Path(sys.argv[1]))
    preparation_set = creation_set(brats_directory)

    output_directory = str(Path(sys.argv[2]))
    
    model = 'it_core_news_sm'
    hprpms_fir_lev = dict(batch_size = 1024, dropout = 0.1, patience = 1000, max_epochs = 21, max_steps = 10000, eps = 0.00000001, learn_rate = 0.001)
    hprpms_sec_lev = dict(batch_size = 1024, dropout = 0.1, patience = 1000, max_epochs = 21, max_steps = 10000, eps = 0.00000001, learn_rate = 0.001)
    hprpms_thi_lev = dict(batch_size = 1024, dropout = 0.2, patience = 1000, max_epochs = 21, max_steps = 10000, eps = 0.00000001, learn_rate = 0.001)
    hprpms_fou_lev = dict(batch_size = 1024, dropout = 0.2, patience = 1000, max_epochs = 21, max_steps = 10000, eps = 0.00000001, learn_rate = 0.001)
    creation_models(preparation_set, os.path.join(output_directory, 'small'), model, hprpms_fir_lev, hprpms_sec_lev, hprpms_thi_lev, hprpms_fou_lev)
    
    model = 'it_core_news_md'
    hprpms_fir_lev = dict(batch_size = 1024, dropout = 0.1, patience = 1000, max_epochs = 21, max_steps = 10000, eps = 0.00000001, learn_rate = 0.001)
    hprpms_sec_lev = dict(batch_size = 1024, dropout = 0.2, patience = 1000, max_epochs = 21, max_steps = 10000, eps = 0.00000001, learn_rate = 0.001)
    hprpms_thi_lev = dict(batch_size = 1024, dropout = 0.1, patience = 1000, max_epochs = 21, max_steps = 10000, eps = 0.00000001, learn_rate = 0.001)
    hprpms_fou_lev = dict(batch_size = 1024, dropout = 0.2, patience = 1000, max_epochs = 21, max_steps = 10000, eps = 0.00000001, learn_rate = 0.001)
    creation_models(preparation_set, os.path.join(output_directory, 'medium'), model, hprpms_fir_lev, hprpms_sec_lev, hprpms_thi_lev, hprpms_fou_lev)
    
    model = 'it_core_news_lg'
    hprpms_fir_lev = dict(batch_size = 1024, dropout = 0.2, patience = 1000, max_epochs = 21, max_steps = 10000, eps = 0.00000001, learn_rate = 0.001)
    hprpms_sec_lev = dict(batch_size = 1024, dropout = 0.1, patience = 1000, max_epochs = 21, max_steps = 10000, eps = 0.00000001, learn_rate = 0.001)
    hprpms_thi_lev = dict(batch_size = 1024, dropout = 0.2, patience = 1000, max_epochs = 21, max_steps = 10000, eps = 0.00000001, learn_rate = 0.001)
    hprpms_fou_lev = dict(batch_size = 1024, dropout = 0.2, patience = 1000, max_epochs = 21, max_steps = 10000, eps = 0.00000001, learn_rate = 0.001)
    creation_models(preparation_set, os.path.join(output_directory, 'large'), model, hprpms_fir_lev, hprpms_sec_lev, hprpms_thi_lev, hprpms_fou_lev)

def process(model_path: str, text_path: str):
    new_nlp = load(model_path)
    new_nlp.remove_pipe('ner')
    text = open(text_path, 'rt', encoding = 'utf-8')
    output_tokens = open(text_path.replace('.txt', '.tokens'), 'wt', encoding = 'utf-8')
    output_tokens.write('paragraph_ID	sentence_ID	token_ID_within_sentence	token_ID_within_document	word	lemma	byte_onset	byte_offset	POS_tag	fine_POS_tag	dependency_relation	syntactic_head	IOB_code_named_entity\n')
    paragraph_id = 0
    sentence_id = 0
    token_id_within_sentence = 0
    token_id_within_document = 0
    byte_onset = 0
    byte_offset = 0
    content_text = text.read()
    i = 0
    doc = new_nlp(content_text)
    for token in doc:
        if '\n' not in str(token.text):
            byte_offset = byte_onset + len(str(token.text))
            i = i + len(str(token.text))
            output_tokens.write(str(paragraph_id) + '	' + str(sentence_id) +  '	' + str(token_id_within_sentence) + '	' + str(token_id_within_document) + '	' + str(token.text) + '	' + str(token.lemma_) + '	' + str(byte_onset) + '	' + str(byte_offset) + '	' + str(token.pos_) + '	' + str(token.tag_) + '	' + str(token.dep_) + '	' + str(token.head) + '	')
            if str(token.ent_iob_) == 'O':
                output_tokens.write(str(token.ent_iob_) + '\n')
            else:
                output_tokens.write(str(token.ent_iob_) + '-' + str(token.ent_type_) + '\n')
            token_id_within_document = token_id_within_document + 1
            byte_onset = byte_offset
            if i < len(content_text):
                if content_text[i] == ' ':
                    while content_text[i] == ' ':
                        i = i + 1
                        byte_onset = byte_onset + 1
            if str(token.text) == '.' or str(token.text) == '!' or str(token.text) == '?' or str(token.text) == '....':
                sentence_id = sentence_id + 1
                token_id_within_sentence = 0
            else:
                token_id_within_sentence = token_id_within_sentence + 1
        elif '\n' in str(token.text):
            if str(token.text) != '\n':
                paragraph_id = paragraph_id + 1
                i = i + len(str(token.text))
                byte_onset = byte_onset + len(str(token.text))
            else:
                i = i + 1
                byte_onset = byte_onset + 1
    text.close()
    output_tokens.close()

if __name__ == '__main__':
    ''' For this function you need to have the first environment variable a path to the dataset in the brat format
    and the second environment variable a path to the output directory'''
    #create_train_evaluation()
    # For this function you need to have one environment variable which is a path to the output_directory to save the new models
    #create_new_models(str(Path(sys.argv[1])))
    ''' For this function you need to have one environment variable which is a path to the directory containing the new model to use
    and a variable which is a path to a directory where to save the processed file'''
    process(sys.argv[1], sys.argv[2])