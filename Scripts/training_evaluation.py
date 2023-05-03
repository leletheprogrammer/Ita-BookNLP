from spacy.cli.train import train
from spacy.cli.evaluate import evaluate
from pathlib import Path

def training_models(output_directory: str):
    train(Path(output_directory + '/first_level/config.cfg'), Path(output_directory + '/first_level/models'), overrides = {'corpora.train.path': str(Path(output_directory + '/first_level/training.spacy')), 'corpora.dev.path': str(Path(output_directory + '/first_level/dev.spacy'))})
    
    train(Path(output_directory + '/second_level/config.cfg'), Path(output_directory + '/second_level/models'), overrides = {'corpora.train.path': str(Path(output_directory + '/second_level/training.spacy')), 'corpora.dev.path': str(Path(output_directory + '/second_level/dev.spacy'))})
    
    train(Path(output_directory + '/third_level/config.cfg'), Path(output_directory + '/third_level/models'), overrides = {'corpora.train.path': str(Path(output_directory + '/third_level/training.spacy')), 'corpora.dev.path': str(Path(output_directory + '/third_level/dev.spacy'))})
    
    train(Path(output_directory + '/fourth_level/config.cfg'), Path(output_directory + '/fourth_level/models'), overrides = {'corpora.train.path': str(Path(output_directory + '/fourth_level/training.spacy')), 'corpora.dev.path': str(Path(output_directory + '/fourth_level/dev.spacy'))})

def evaluate_models(output_directory: str):
    scores_first_model = evaluate(Path(output_directory + '/first_level/models/model-best'), Path(output_directory + '/first_level/test.spacy'), Path(output_directory + '/first_level/evaluation.json'))
    print('The precision, recall and f1-score of the first model are: ' + str(scores_first_model['ents_p']) + ', ' + str(scores_first_model['ents_r']) + ', ' + str(scores_first_model['ents_f']))
    
    scores_second_model = evaluate(Path(output_directory + '/second_level/models/model-best'), Path(output_directory + '/second_level/test.spacy'), Path(output_directory + '/second_level/evaluation.json'))
    print('The precision, recall and f1-score of the second model are: ' + str(scores_second_model['ents_p']) + ', ' + str(scores_second_model['ents_r']) + ', ' + str(scores_second_model['ents_f']))
    
    scores_third_model = evaluate(Path(output_directory + '/third_level/models/model-best'), Path(output_directory + '/third_level/test.spacy'), Path(output_directory + '/third_level/evaluation.json'))
    print('The precision, recall and f1-score of the third model are: ' + str(scores_third_model['ents_p']) + ', ' + str(scores_third_model['ents_r']) + ', ' + str(scores_third_model['ents_f']))
    
    scores_fourth_model = evaluate(Path(output_directory + '/fourth_level/models/model-best'), Path(output_directory + '/fourth_level/test.spacy'), Path(output_directory + '/fourth_level/evaluation.json'))
    print('The precision, recall and f1-score of the fourth model are: ' + str(scores_fourth_model['ents_p']) + ', ' + str(scores_fourth_model['ents_r']) + ', ' + str(scores_fourth_model['ents_f']))