import librosa
import tensorflow as tf
import numpy as np

SAVED_MODEL_PATH = "models/LSTM_weights2.h5"
SAMPLES_TO_CONSIDER = 16000

class _Keyword_Spotting_Service:

    model = None
    _mapping = [
        "abnormal",
        "normal"
    ]
    _instance = None


    def predict(self, file_path):

        # extract MFCC
        MFCCs,signal = self.preprocess(file_path)

        # we need a 4-dim array to feed to the model for prediction: (# samples, # coefficients, 1)
        MFCCs = MFCCs[np.newaxis,...]
        
        # get the predicted label
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        return (predicted_keyword,signal)


    def preprocess(self, file_path, num_mfcc=40):

        # load audio file
        signal, sample_rate = librosa.load(file_path)

        if len(signal) >= SAMPLES_TO_CONSIDER:
            # ensure consistency of the length of the signal
            signal = signal[:SAMPLES_TO_CONSIDER*12]

            # extract MFCCs
            MFCCs = np.mean(librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc).T,axis=0) 
            MFCCs = np.array(MFCCs).reshape([-1,1])
            print("shape:MFCCs",MFCCs.shape)
        return (MFCCs,signal)



def Keyword_Spotting_Service():
    # ensure an instance is created only the first time the factory function is called
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    return _Keyword_Spotting_Service._instance



if __name__ == "__main__":

    # create 2 instances of the keyword spotting service
    kss = Keyword_Spotting_Service()
    kss1 = Keyword_Spotting_Service()

    # check that different instances of the keyword spotting service point back to the same object (singleton)
    assert kss is kss1
