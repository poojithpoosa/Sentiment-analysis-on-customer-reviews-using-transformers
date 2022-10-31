import pandas as pd
import matplotlib.pyplot as plt
import string 
import numpy as np
from string import digits

data=pd.read_csv('train.csv')
data_valid=pd.read_csv('valid.csv')
data_test=pd.read_csv('test.csv')

print(data['sentiment'].value_counts())
plt.title('Training set labels count')
plt.bar(data['sentiment'].unique(),data['sentiment'].value_counts())
plt.show()

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

for label in data['sentiment'].unique():
    print(label)
    c_words = ''
    stopword = set(STOPWORDS)
    df=data[data['sentiment']==label]
    for words in df['review']:
        words = str(words)
        tks = words.split()
        for i in range(len(tks)):
            tks[i] = tks[i].lower()
        c_words += " ".join(tks)+" "
     
    wordcloud = WordCloud(width = 1000, height = 800,
                    background_color ='white',
                    stopwords = stopword,
                    min_font_size = 10).generate(c_words)
                   
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()

def preprocess_string(data):
    data1=[]
    for i in range(len(data)):
            lower=data[i].lower()
            no_punchuation = lower.translate(str.maketrans('', '', string.punctuation))
            remove_digits = str.maketrans('', '', digits)
            res = no_punchuation.translate(remove_digits)
            no_white_spaces=res.strip()
            data1.append(no_white_spaces)
            
    return np.asarray(data1)

x_train=preprocess_string(data['review'].values)
x_valid=preprocess_string(data_valid['review'].values)
x_test=preprocess_string(data_test['review'].values)
y_train=data['sentiment'].values
y_valid=data_valid['sentiment'].values
y_test=data_test['sentiment'].values


from TextFeatureSelection import TextFeatureSelection


input_doc_list=list(x_train)
target=list(y_train)
feature_selectionOBJ=TextFeatureSelection(target=target,input_doc_list=input_doc_list,metric_list='IG')
result_df=feature_selectionOBJ.getScore()
print(result_df)

feature=result_df[result_df['Information Gain']==0]
feature=feature['word list'].tolist()

def feature_selection(feature,data):
    total=[]
    for i in data:  
        query = i
        querywords = query.split()
        resultwords  = [word for word in querywords if word.lower() not in feature]
        result = ' '.join(resultwords)
        total.append(result)
    return np.array(total)

x_train=feature_selection(feature, x_train)
x_valid=feature_selection(feature,x_valid)
x_test=feature_selection(feature, x_test)
          

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder().fit(y_train.reshape(-1,1))

y_train=enc.transform(y_train.reshape(-1,1)).toarray()
y_valid=enc.transform(y_valid.reshape(-1,1)).toarray()
y_test=enc.transform(y_test.reshape(-1,1)).toarray()


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(x_train)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(x_train)
pad_train = pad_sequences(sequences, padding='post',maxlen=180)
reverse_word_index = tokenizer.index_word

sequences = tokenizer.texts_to_sequences(x_valid)
pad_valid = pad_sequences(sequences, padding='post',maxlen=180)

sequences = tokenizer.texts_to_sequences(x_test)
pad_test = pad_sequences(sequences, padding='post',maxlen=180)


class TransBlock(layers.Layer):
    def __init__(self, emb_dim, num_heads, feed_dim):
        super(TransBlock, self).__init__()
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=emb_dim)
        self.feed_net = keras.Sequential(
            [
                layers.Dense(feed_dim, activation="relu"),
                layers.Dense(emb_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(0.01)
        self.dropout2 = layers.Dropout(0.01)

    def call(self, inputs, training):
        attention_output = self.attention(inputs, inputs)
        attention_output = self.dropout1(attention_output, training=training)
        out1 = self.layernorm1(inputs + attention_output)
        feed_output = self.feed_net(out1)
        feed_output = self.dropout2(feed_output, training=training)
        return self.layernorm2(out1 + feed_output)


class PosEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(PosEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, out):
        maxlen = tf.shape(out)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        out = self.token_emb(out)
        return out + positions


inputs = layers.Input(shape=(180,))
embedding_layer = PosEmbedding(180, 12000 , 32)
layer = embedding_layer(inputs)
transformer_block = TransBlock(32, 3, 32)
layer = transformer_block(layer)
layer = layers.Dropout(0.1)(layer)
layer = transformer_block(layer)
layer = layers.GlobalAveragePooling1D()(layer)
layer = layers.Dropout(0.1)(layer)
layer= layers.Dense(20, activation="relu")(layer)
outputs = layers.Dense(2, activation="softmax")(layer)
model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())


model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=[["accuracy"],[tf.keras.metrics.Precision()]]
)
history = model.fit(
    pad_train, y_train, batch_size=32, epochs=10,validation_data=(pad_valid,y_valid)
)

result=model.evaluate(pad_test,y_test)
result2=model.evaluate(pad_train,y_train)

acc_train=history.history['accuracy']
acc_val=history.history['val_accuracy']
plt.title('model accuracy')
plt.plot(acc_train)
plt.plot(acc_val)
plt.xlabel('Epochs')
plt.ylabel('accuarcy')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


loss_train=history.history['loss']
loss_val=history.history['val_loss']
plt.title('model loss')
plt.plot(loss_train)
plt.plot(loss_val)
plt.xlabel('loss')
plt.ylabel('loss')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


pre_train=history.history['precision']
pre_val=history.history['val_precision']
plt.title('model precision')
plt.plot(pre_train)
plt.plot(pre_val)
plt.xlabel('loss')
plt.ylabel('precision')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


def new_predction(data):
    data1=[]
    for i in range(len(data)):
        lower=data[i].lower()
        no_punchuation = lower.translate(str.maketrans('', '', string.punctuation))
        no_white_spaces=no_punchuation.strip()
        data1.append(no_white_spaces)

    sequences = tokenizer.texts_to_sequences(data1)

    for i in range(len(sequences)):
        if len(sequences[i])>180:
            t=sequences[i]
            sequences[i]=t[:180]
        else:
            lenght=len(sequences[i])
            res=180-lenght

            for j in range(res):
                sequences[i].append(0)
    y_pred=model.predict(sequences)
    labels=['negative','positive']
    for i in y_pred:
        print(labels[np.argmax(i,axis=0)])

new_predction(['worst product',' good'])
