from datasets import load_dataset

class DataSetter:
  def __init__(self, dataset_name):
    self.dataset_name = dataset_name
    self.dataset = self.load_dataset()

  def load_dataset(self):
    return load_dataset(self.dataset_name)
  
  def get_dataset(self):
    return self.dataset
  
