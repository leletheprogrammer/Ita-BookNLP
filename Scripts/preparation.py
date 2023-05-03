import os
from spacy import Language, load
from saving_examples import split_save, initialize_config_file

def creation_set(brats_directory: str):
    preparation_set = list()

    file_names = os.listdir(brats_directory)
    
    num_first_level = 0
    num_second_level = 0
    num_third_level = 0
    num_fourth_level = 0

    i = 0
    while i < len(file_names):
        annotation_file = open(os.path.join(brats_directory, file_names[i]), 'rt', encoding = 'utf-8')
        first_level_annotations = list()
        second_level_annotations = list()
        third_level_annotations = list()
        fourth_level_annotations = list()
        for line in annotation_file.readlines():
            annotation = line.split('	')[1].split(' ')
            if first_level_annotations and int(annotation[1]) < first_level_annotations[-1][1]:
                if second_level_annotations and int(annotation[1]) < second_level_annotations[-1][1]:
                    if third_level_annotations and int(annotation[1]) < third_level_annotations[-1][1]:
                        fourth_level_annotations.append(tuple((int(annotation[1]), int(annotation[2]), annotation[0])))
                        num_fourth_level = num_fourth_level + 1
                    else:
                        third_level_annotations.append(tuple((int(annotation[1]), int(annotation[2]), annotation[0])))
                        num_third_level = num_third_level + 1
                else:
                    second_level_annotations.append(tuple((int(annotation[1]), int(annotation[2]), annotation[0])))
                    num_second_level = num_second_level + 1
            else:
                first_level_annotations.append(tuple((int(annotation[1]), int(annotation[2]), annotation[0])))
                num_first_level = num_first_level + 1
        annotation_file.close()
        text_file = open(os.path.join(brats_directory, file_names[i + 1]), 'rt', encoding = 'utf-8')
        preparation_set.append(tuple((text_file.read(), first_level_annotations, second_level_annotations, third_level_annotations, fourth_level_annotations)))
        text_file.close()
        i = i + 2
    
    print('There are ' + str(num_first_level) + ' annotations of the first level')
    print('There are ' + str(num_second_level) + ' annotations of the second level')
    print('There are ' + str(num_third_level) + ' annotations of the third level')
    print('There are ' + str(num_fourth_level) + ' annotations of the fourth level')
    
    return preparation_set

def creation_document(nlp: Language, text: str, annotations: list):
    document = nlp(text)
    ents = list()
    for start, end, label in annotations:
        span = document.char_span(start, end, label = label)
        ents.append(span)
    document.ents = ents
    
    return document

def preprocessing(nlp: Language, preparation_set: list):
    documents_first_level = list()
    documents_second_level = list()
    documents_third_level = list()
    documents_fourth_level = list()

    for text, first_level_annotations, second_level_annotations, third_level_annotations, fourth_level_annotations in preparation_set:
        
        documents_first_level.append(creation_document(nlp, text, first_level_annotations))
        
        documents_second_level.append(creation_document(nlp, text, second_level_annotations))
        
        documents_third_level.append(creation_document(nlp, text, third_level_annotations))
        
        documents_fourth_level.append(creation_document(nlp, text, fourth_level_annotations))
    
    return documents_first_level, documents_second_level, documents_third_level, documents_fourth_level

def preparation_training(preparation_set: list, output_directory: str, model: str, hprpms_fir_lev: dict, hprpms_sec_lev: dict, hprpms_thi_lev: dict, hprpms_fou_lev: dict):
    nlp = load(model)
    
    documents_first_level, documents_second_level, documents_third_level, documents_fourth_level = preprocessing(nlp, preparation_set)
    
    split_save(documents_first_level, os.path.join(output_directory, 'first_level'))
    initialize_config_file(os.path.join(output_directory, 'first_level'), hprpms_fir_lev['batch_size'], hprpms_fir_lev['dropout'], hprpms_fir_lev['patience'], hprpms_fir_lev['max_epochs'], hprpms_fir_lev['max_steps'], hprpms_fir_lev['eps'], hprpms_fir_lev['learn_rate'])
    
    split_save(documents_second_level, os.path.join(output_directory, 'second_level'))
    initialize_config_file(os.path.join(output_directory, 'second_level'), hprpms_sec_lev['batch_size'], hprpms_sec_lev['dropout'], hprpms_sec_lev['patience'], hprpms_sec_lev['max_epochs'], hprpms_sec_lev['max_steps'], hprpms_sec_lev['eps'], hprpms_sec_lev['learn_rate'])
    
    split_save(documents_third_level, os.path.join(output_directory, 'third_level'))
    initialize_config_file(os.path.join(output_directory, 'third_level'), hprpms_thi_lev['batch_size'], hprpms_thi_lev['dropout'], hprpms_thi_lev['patience'], hprpms_thi_lev['max_epochs'], hprpms_thi_lev['max_steps'], hprpms_thi_lev['eps'], hprpms_thi_lev['learn_rate'])
    
    split_save(documents_fourth_level, os.path.join(output_directory, 'fourth_level'))
    initialize_config_file(os.path.join(output_directory, 'fourth_level'), hprpms_fou_lev['batch_size'], hprpms_fou_lev['dropout'], hprpms_fou_lev['patience'], hprpms_fou_lev['max_epochs'], hprpms_fou_lev['max_steps'], hprpms_fou_lev['eps'], hprpms_fou_lev['learn_rate'])