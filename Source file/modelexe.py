import keras
import re
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
load_model=keras.models.load_model("modellstm.h5")
#@title Interface { run: "auto", vertical-output: true, display-mode: "form" }
print('\n\n\n\n\n\n LOADING.......')
tokenizer = pickle.load(open('tokenizer.pickle','rb'))



test = input("Input The Message: ")


test=[(test)]
seq = tokenizer.texts_to_sequences(test)
padded = sequence.pad_sequences(seq, maxlen=500)
pred = load_model.predict(padded)[0]
print("pred", pred)

if pred <0.5:
  
  print("Message marked as ham")
    
else:
  
  print("Message marked a spam")


x= input("Please press Enter")
