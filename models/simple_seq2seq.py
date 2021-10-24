from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
from sklearn.metrics import f1_score


class SimpleSeq2Seq:
    def __init__(self, encoder_type, encoder_name, decoder_name, num_epochs, train_path, eval_path):
        self.encoder_type = encoder_type

        self.model = Seq2SeqModel(encoder_type, encoder_name, decoder_name)

        self.args = {
            "num_train_epochs": num_epochs
        }

        self.train_path = train_path
        self.eval_path = eval_path

    def train(self, data_train, data_val):
        self.model.train_model(data_train, output_dir=self.train_path, args=self.args, eval_data=data_val, f1=f1_score)

    def evaluate(self, data_test):
        self.model.eval_model(data_test, output_dir=self.eval_path, f1=f1_score)

    def predict(self, sentence_list):
        self.model = Seq2SeqModel(self.encoder_type, self.train_path, self.train_path)
        predictions = self.model.predict(sentence_list)
        return predictions




