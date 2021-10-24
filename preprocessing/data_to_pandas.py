from preprocessing.data_to_sequence import *
import pandas as pd
import constants


def qa_to_pandas(filein, fileout):
    if constants.IS_REBUILD == 1:
        convo = daily_conversations(filein)
        questions, answers = conversation_to_qa(convo)

        questions = pd.DataFrame(questions, columns=['input_text'])
        answers = pd.DataFrame(answers, columns=['target_text'])

        data = pd.concat([questions, answers], axis=1)

        data.to_csv(fileout, index=False)

    else:
        data = pd.read_csv(fileout)

    return data




