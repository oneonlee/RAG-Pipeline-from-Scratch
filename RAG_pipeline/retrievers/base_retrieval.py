class BaseRetriever:
    def __init__(self):
        pass
        
    def retrieve(self, sequence_input_ids, dataset, k=1):
        raise NotImplementedError
