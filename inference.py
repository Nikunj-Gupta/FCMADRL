from framework import FCMADRL
from config import *


class Inference:
    def __init__(self, path=FILE_PATH_DQN, file_name=FILE_NAME_DQN):
        self.path = path
        self.file_name = file_name
        self.mas = FCMADRL()
        self.dqn = self.mas.get_dqn()
        self.model = self.mas.get_dqn_model(self.dqn)

    def model_load(self):
        arg = self.path + self.file_name
        self.model.load_weights(arg)
        self.mas.use_existing_dqn(self.model)
        total_reward = self.mas.fcmadrl()
        return total_reward


if __name__ == "__main__":
    inf = Inference()
    inf = Inference(file_name="model5.h5")
    tr = inf.model_load()
    print tr
