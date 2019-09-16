from rasa_nlu import config
from rasa_nlu.model import Interpreter
from rasa_nlu.model import Trainer
from rasa_nlu.training_data import load_data


def train (data, config_file, model_dir):
    training_data = load_data(data)
    trainer = Trainer(config.load(config_file))
    trainer.train(training_data,num_threads=3)
    trainer.persist(model_dir, fixed_model_name='chat')

train('data/nlu.md', 'nlu_config.yml', 'models/nlu')
interpreter = Interpreter.load('./models/nlu/default/chat')

# define function to ask question
def ask_question(text):
    print(interpreter.parse(text))

ask_question('hello em')
ask_question('hi')