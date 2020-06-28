import pandas as pd
import tensorflow as tf
import json
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Input, Dense,Concatenate,ReLU,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_auc_score,explained_variance_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel
from scipy.special import expit

from mmoe import MMoE
import random

random.seed(44)

SEED=1
# Simple callback to print out ROC-AUC, only for classification
class ROCCallback(Callback):
    def __init__(self, training_data, validation_data, test_data):
        self.train_X = training_data[0]
        self.train_Y = training_data[1]
        self.validation_X = validation_data[0]
        self.validation_Y = validation_data[1]
        self.test_X = test_data[0]
        self.test_Y = test_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        train_prediction = self.model.predict(self.train_X)
        validation_prediction = self.model.predict(self.validation_X)
        test_prediction = self.model.predict(self.test_X)

        # Iterate through each task and output their ROC-AUC across different datasets
        for index, output_name in enumerate(self.model.output_names):
            if (output_name == 'user_click') or (output_name == 'user_like'): # cassification is roc-auc score
                train_roc_auc = roc_auc_score(self.train_Y[index], np.squeeze(train_prediction[index]))
                validation_roc_auc = roc_auc_score(self.validation_Y[index], np.squeeze(validation_prediction[index]))
                test_roc_auc = roc_auc_score(self.test_Y[index], np.squeeze(test_prediction[index]))
                print(
                    'ROC-AUC-{}-Train: {} ROC-AUC-{}-Validation: {} ROC-AUC-{}-Test: {}'.format(
                        output_name, round(train_roc_auc, 4),
                        output_name, round(validation_roc_auc, 4),
                        output_name, round(test_roc_auc, 4)
                    )
                )
            elif (output_name == 'user_rating') or (output_name == 'time_spend'): # regressio is explained variance
                train_roc_auc = explained_variance_score(self.train_Y[index], np.squeeze(train_prediction[index]))
                validation_roc_auc = explained_variance_score(self.validation_Y[index], np.squeeze(validation_prediction[index]))
                test_roc_auc = explained_variance_score(self.test_Y[index], np.squeeze(test_prediction[index]))
                print(
                    'explained-variance-score-{}-Train: {} explained-variance-score-{}-Validation: {} explained-variance-score-{}-Test: {}'.format(
                        output_name, round(train_roc_auc, 4),
                        output_name, round(validation_roc_auc, 4),
                        output_name, round(test_roc_auc, 4)
                    )
                )

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


#### =====================
### input data
#### =====================

def manipulate_data():
    with open('data/training_set.json') as jsonfile:
        data = json.load(jsonfile)

    df = pd.DataFrame(data)
    df = df.reset_index().rename(columns={'index':'videoID'})
    df = df.drop(columns=['label','videoID','channel_title'])

    ## generate 20 users
    print("generating 20 users...")
    df['userId'] = np.random.randint(1, 21, df.shape[0])

    # manipulate the data
    print("manipulating user click and user rating..")
    df['user_click'] = np.random.randint(0, 2, df.shape[0])
    df['user_rating'] = np.random.randint(0, 6, df.shape[0])
    print("manupulating user like and time sepend on each video ...")
    df['user_like'] = np.random.randint(0, 2, df.shape[0])
    df['time_spend'] = np.random.randint(0, 11, df.shape[0])
    df.loc[df['user_click'] == 0, 'user_like'] = 0
    df.loc[df['user_click'] == 0, 'time_spend'] = 0

    # print(df.columns)
    # Index(['title', 'view_count', 'tags', 'description', 'userId', 'user_click',
    #        'user_rating', 'user_like', 'time_spend'],
    #       dtype='object')

    # cold start: ranking by popularility (view_count)
    print("manupulating recommendated video ranking position (sorted by view count)...")
    df['view_count'] = df['view_count'].fillna(0).astype(int)
    df['position'] = df.groupby("userId")["view_count"].rank(ascending=False)

    df['pos_bias'] = df['user_click']+1
    df.loc[df['pos_bias'] == 2, 'pos_bias'] = 0

    # missing feature in paper: device info
    print("manuipulate device info...")
    df['device_info'] = [random.choice(['ios','android','web']) for _ in range(0,df.shape[0])]

    ## creating bert features for text features

    # corresponding to paper : Embedding for query and candidate items (from query and caxdidrae video features; user and context features)
    # get text embeddings
    print("generating bert embedding videos ....")
    df['tags'] = df['tags'].fillna(' ')
    df['tags'] = df['tags'].apply(lambda x: ','.join(x) if isinstance(x,list)==True else x)
    # combine all video info into one embeddings
    df['video_emb'] = df['title']+ df['description']

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = TFBertModel.from_pretrained('bert-base-cased')
    max_length = 10 # change it to your computer capacity
    batch_encoding = tokenizer.batch_encode_plus(df['video_emb'].tolist(), max_length=max_length, pad_to_max_length=True)

    outputs = model(tf.convert_to_tensor(batch_encoding['input_ids'])) # shape: (batch,sequence length, hidden state)
    embeddings_video = tf.reduce_mean(outputs[0],1)
    df['video_emb'] = embeddings_video.numpy().tolist()
    # df['video_emb'] = df['video_emb'].apply(lambda x: np.array(x,dtype=np.float))

    df = df.drop(columns=['title','description'])
    print("generating bert embedding for user...")
    # corresponding to paper: Embeddings for visual and language, and context features (from user and context features)
    # get user embeddings
    # assuming tags as user interested tags
    batch_encoding_user = tokenizer.batch_encode_plus(df['tags'].tolist(), max_length=max_length, pad_to_max_length=True)
    outputs_user = model(tf.convert_to_tensor(batch_encoding_user['input_ids'])) # shape: (batch,sequence length, hidden state)
    embeddings_user = tf.reduce_mean(outputs_user[0],1)
    df['user_emb'] = embeddings_user.numpy().tolist()

    # to speedup:
    df = df.reset_index(drop=True) #shuffle df

    # missing features in paper: dense features (from query and caxdidrae video features; user and context features)
    return df

#### ==================
## train the model
#### ==================
def train_ranking_model(df):

    train, val_test = train_test_split(df, test_size=0.3)
    val, test = train_test_split(val_test, test_size=0.5)


    train_label = [train[col_name].values for col_name in ['user_click', 'user_rating', 'user_like', 'time_spend']]
    # Tensorflow - ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type float)
    train_data = [np.asarray(np.squeeze(train[['video_emb']].values.tolist())).astype(np.float32),
                  np.asarray(np.squeeze(train[['user_emb']].values.tolist())).astype(np.float32),
                  train[['view_count']].values] # todo: user demographics, device, time, and location
                  # np.asarray(train[['view_count']].values.tolist()).astype(np.float32)]

    validation_label = [val[col_name].values for col_name in ['user_click', 'user_rating', 'user_like', 'time_spend']]
    validation_data = [np.asarray(np.squeeze(val[['video_emb']].values.tolist())).astype(np.float32),
                       np.asarray(np.squeeze(val[['user_emb']].values.tolist())).astype(np.float32),
                       val[['view_count']].values]
                  # np.asarray(val[['view_count']].values.tolist()).astype(np.float32)]

    test_label = [test[col_name].values for col_name in ['user_click', 'user_rating', 'user_like', 'time_spend']]
    test_data = [np.asarray(np.squeeze(test[['video_emb']].values.tolist())).astype(np.float32),
                 np.asarray(np.squeeze(test[['user_emb']].values.tolist())).astype(np.float32),
                 test[['view_count']].values]
                  # np.asarray(test[['view_count']].values.tolist()).astype(np.float32)]

    print("Output is user_click, user_rating, user_like and time_spend...")
    output_info = [(1, 'user_click'),(1,'user_rating'),(1,'user_like'),(1,'time_spend')]  # the rating is categorical or regression?

    output_activation = ['softmax','linear','softmax','linear'] # None (linear) activation for regression task; softmax for classification

    print('Training data shape = {}'.format(train.shape))
    print('Validation data shape = {}'.format(val.shape))
    print('Test data shape = {}'.format(test.shape))

    # Set up the input layer
    input_video_emb = Input(shape=(768,))
    input_user_emb = Input(shape=(768,))
    input_other_features = Input(shape=(1,))
    input = Concatenate()([input_video_emb,input_user_emb,input_other_features])
    input_layer = ReLU()(input)

    # add the shared ReLu layer
    # Set up MMoE layer
    mmoe_layers = MMoE(
        units=4,
        num_experts=8,
        num_tasks=4
    )(input_layer)

    output_layers = []

    # Build tower layer from MMoE layer
    for index, task_layer in enumerate(mmoe_layers):
        tower_layer = Dense(
            units=8,
            activation='relu',
            kernel_initializer=VarianceScaling())(task_layer)
        output_layer = Dense(
            units=output_info[index][0],
            name=output_info[index][1],
            activation=output_activation[index],
            kernel_initializer=VarianceScaling())(tower_layer)
        output_layers.append(output_layer)

    # Compile model
    # model = Model(inputs=[input_video_tags,input_video_title,input_video_desp,input_video_view], outputs=output_layers)
    model = Model(inputs=[input_video_emb,input_user_emb,input_other_features], outputs=output_layers)
    adam_optimizer = Adam()
    model.compile(
        loss={'user_click': 'binary_crossentropy', 'user_rating':'MSE','user_like': 'binary_crossentropy','time_spend':'MSE'},
        optimizer=adam_optimizer,
        metrics=['accuracy']
    )

    # Print out model architecture summary
    model.summary()

# Train the model,
    model.fit(
        x=train_data,
        y=train_label,
        validation_data=(validation_data, validation_label),
        callbacks=[
            ROCCallback(
                training_data=(train_data, train_label),
                validation_data=(validation_data, validation_label),
                test_data=(test_data, test_label)
            )
        ],
        epochs=100
    )

    return model


def train_position_bias_model(df):
    ###  ============================
    ## adding a shallow side tower to learn selection biase
    # measure https://en.wikipedia.org/wiki/Propensity_score_matching
    #### =============================
    ## train the selection bias
    ## "shallow tower": input: item position; output: relevance (clicked or not);

    pos_shallow_tower = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='softmax')
        ]
    )

    print("the network structure for position bias prediction:",pos_shallow_tower.summary())
    # sgd_opt = tf.keras.optimizers.SGD(lr=0.001)
    pos_shallow_tower.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )


    position_df = pd.get_dummies(df, columns=['device_info'], prefix='', prefix_sep='')[['android', 'ios', 'web','pos_bias','position']]
    assert position_df.isnull().any().any() == False

    train_pos, val_pos = train_test_split(position_df, test_size=0.3)
    train_pos_data = train_pos.drop(columns={'pos_bias'})
    train_pos_label = train_pos[['pos_bias']]

    validation_pos_data =  val_pos.drop(columns={'pos_bias'})
    validation_pos_label = val_pos[['pos_bias']]

    pos_shallow_tower.fit(
        x=train_pos_data,
        y=train_pos_label,
        validation_data=(validation_pos_data, validation_pos_label),
        epochs=10
    )
    return pos_shallow_tower

    ## todo: in real data, check the bias weather increase as the position goes up

#### =================
## combine prediction
#### =================
# final score for each candidate videos =  w1* sigmoid (logit for user engagament + logit for selection biase) + w2* user satisfaction

def final_score(weights_for_engagement, weights_for_satification, main_model, position_biase_model,test_main,test_position):

    print("the manually set weights for user engagement is", str(weights_for_engagement))
    print("the manually set weights for user satisfaction is", str(weights_for_satification))

    preds = main_model.predict(test_main)
    user_click = preds[0]
    user_rating = preds[1]
    user_like = preds[2]
    time_spend = preds[3]

    preds_position = position_biase_model.predict(test_position)

    return weights_for_engagement*expit(user_click+time_spend+preds_position) + weights_for_satification*expit(user_rating+user_like)



def main():
    print("Manipulating data....")
    df = manipulate_data()
    print("train the main model ...")
    main_model = train_ranking_model(df)
    print("train the shallow tower for position bias...")
    position_biase_model = train_position_bias_model(df)

    print("testing data...")
    test_data = df.sample(frac=0.1)
    test_main = [np.asarray(np.squeeze(test_data[['video_emb']].values.tolist())).astype(np.float32),
              np.asarray(np.squeeze(test_data[['user_emb']].values.tolist())).astype(np.float32),
              test_data[['view_count']].values]

    test_position = pd.get_dummies(test_data, columns=['device_info'], prefix='', prefix_sep='')[['android', 'ios', 'web','position']]
    print("The final score for test data is...")
    print(final_score(0.2, 0.8, main_model, position_biase_model,test_main,test_position))

if __name__ == "__main__":
    main()
