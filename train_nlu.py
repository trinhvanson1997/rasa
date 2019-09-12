
from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config
from rasa_nlu.model import Metadata, Interpreter
import json

def train (data, config_file, model_dir):
    training_data = load_data(data)
    trainer = Trainer(config.load(config_file))
    trainer.train(training_data)
    trainer.persist(model_dir, fixed_model_name = 'chat')

train('nlu.md', 'nlu_config.yml', 'models/nlu')
interpreter = Interpreter.load('./models/nlu/default/chat')

# define function to ask question
def ask_question(text):
    print(interpreter.parse(text))

ask_question('em sinh ngày bao nhiêu')
ask_question('cho anh số điện thoại đi')