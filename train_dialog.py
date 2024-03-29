import logging

from rasa_core.agent import Agent
from rasa_core.policies import FallbackPolicy
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy


# Function
# ------------
def train_dialog(dialog_training_data_file, domain_file, path_to_model='models/dialog'):
    logging.basicConfig(level='INFO')
    fallback = FallbackPolicy(fallback_action_name="utter_unclear", core_threshold=0.3, nlu_threshold=0.3)

    agent = Agent(domain_file,
                  policies=[MemoizationPolicy(max_history=1), KerasPolicy(epochs=200,
                                                                          batch_size=20), fallback])
    training_data = agent.load_data(dialog_training_data_file)
    agent.train(
        training_data,
        augmentation_factor=50,
        validation_split=0.2)
    agent.persist(path_to_model)


# Train
# --------
train_dialog('data/stories.md', 'domain.yml')
