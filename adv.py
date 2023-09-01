from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn import preprocessing
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python.keras.utils import np_utils
# from keras.utils import np_utils

from util import gen_tf2_bim, gen_tf2_fgsm_attack, gen_tf2_pgm, gen_tf2_mim

np.random.seed(10)

COL_NAME = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
            'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
            'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

NUMERIC_COLS = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment',
                'urgent', 'hot', 'num_failed_logins', 'num_compromised',
                'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                'num_shells', 'num_access_files', 'num_outbound_cmds', 'count',
                'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
                'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']
#
# class Autoencoder(Model):
#     def __init__(self, latent_dim, input_size, num_of_class):
#         super(Autoencoder, self).__init__()
#
#         print("input_size: ", input_size)
#
#         self.input_size = input_size
#         self.num_of_class = num_of_class
#         self.latent_dim = latent_dim
#         self.encoder = tf.keras.Sequential([
#         layers.Flatten(), layers.Dense(latent_dim, activation='relu'),])
#         self.decoder = tf.keras.Sequential([
#             layers.Dense(200, input_dim=input_size, activation=tf.nn.relu),
#             layers.Dense(500, activation=tf.nn.relu),
#             layers.Dense(200, activation=tf.nn.relu),
#             layers.Dense(num_of_class),
#             layers.Activation(tf.nn.softmax)
#             ])
#
#     def call(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded

# class Denoise(Model):
#   def __init__(self):
#     super(Denoise, self).__init__()
#     self.encoder = tf.keras.Sequential([
#       layers.Input(shape=(28, 28, 1)),
#       layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
#       layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])
#
#     self.decoder = tf.keras.Sequential([
#       layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
#       layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
#       layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])
#
#   def call(self, x):
#     encoded = self.encoder(x)
#     decoded = self.decoder(encoded)
#     return decoded





def main():
    EPOCH = 50
    TEST_RATE = 0.2
    VALIDATION_RATE = 0.2

    X, y = get_ds()

    num_class = len(np.unique(y))

    attack_functions = [gen_tf2_fgsm_attack, gen_tf2_bim, gen_tf2_pgm, gen_tf2_mim]

    print(X.shape[1])

    # print("OKKKKKKK")
    # exit()

    model = create_tf_model(X.shape[1], num_class)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_RATE)
    y_train_cat = np_utils.to_categorical(y_train)
    y_test_cat = np_utils.to_categorical(y_test)

    history = model.fit(X_train, y_train_cat, epochs=EPOCH,
                        batch_size=50000, verbose=0,
                        validation_split=VALIDATION_RATE)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm_org = confusion_matrix(y_test, y_pred)
    print("_" * 50)
    print("Original confusion matrix")
    print(cm_org)
    evaluation_metrics(y_test, y_pred)
    print("_" * 50)
    # exit()
    for attack_function in attack_functions:
        print("<"*15, attack_function, ">"*15)
        # epsilon = 0.1
        # while epsilon < 1:
        for epsilon in [.1, .5, 1, 1.5, 2, 2.5, 3]:
            print("-" * 35)
            print("eps: ", epsilon)
            model = create_tf_model(X.shape[1], num_class)
            history = model.fit(X_train, y_train_cat, epochs=EPOCH,
                                batch_size=50000, verbose=0,
                                validation_split=VALIDATION_RATE)
            adv_x = attack_function(model, X_test, epsilon)
            #
            # latent_dim = 64
            # autoencoder = Autoencoder(latent_dim, X.shape[1], num_class)
            # autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
            # autoencoder.fit(X_train, y_train_cat,
            #                 epochs=EPOCH, batch_size=50000, verbose=0,
            #                 validation_split=VALIDATION_RATE)
            # print("Encoder done !")

            y_pred = np.argmax(model.predict(adv_x), axis=1)
            cm_adv = confusion_matrix(y_test, y_pred)


            print("Attacked confusion matrix")

            print(cm_adv)
            evaluation_metrics(y_test, y_pred)

            # adv_x_test = autoencoder.decoder(adv_x)
            # print("decoder done !")


            print("Adversarial training ", "."*15)
            # define the checkpoint
            adv_x_training = attack_function(model, X_train, epsilon)
            adv_x_test = attack_function(model, X_test, epsilon)

            concat_adv_x = np.concatenate([X_train, adv_x_training])
            concat_y_train = np.concatenate([y_train_cat, y_train_cat])

            history = model.fit(concat_adv_x, concat_y_train, epochs=EPOCH,
                                batch_size=50000, verbose=0,
                                validation_data=(adv_x_test, y_test_cat))

            y_pred = np.argmax(model.predict(adv_x_test), axis=1)
            cm_adv = confusion_matrix(y_test, y_pred)
            print("Attacked confusion matrix - adv training ")
            print(cm_adv)
            evaluation_metrics(y_test, y_pred)

            epsilon += .1


def evaluation_metrics(y_test, y_pred):
    print("accuracy_score: ", accuracy_score(y_test, y_pred))
    print("recall_score: ", recall_score(y_test, y_pred, average='weighted'))
    print("precision_score: ", precision_score(y_test, y_pred, average='weighted'))
    print("f1_score: ", f1_score(y_test, y_pred, average='weighted'))

def get_ds():
    """ get_ds: Get the numeric values of the KDDCUP'99 dataset. """
    x_kddcup, y_kddcup = fetch_kddcup99(return_X_y=True, shuffle=False)
    df_kddcup = pd.DataFrame(x_kddcup, columns=COL_NAME)
    df_kddcup['label'] = y_kddcup
    df_kddcup.drop_duplicates(keep='first', inplace=True)
    df_kddcup['label'] = df_kddcup['label'].apply(lambda d: \
                                    str(d).replace('.', '').replace("b'", "").\
                                        replace("'", ""))

    conversion_dict = {'back':'dos', 'buffer_overflow':'u2r', 'ftp_write':'r2l',
                       'guess_passwd':'r2l', 'imap':'r2l', 'ipsweep':'probe',
                       'land':'dos', 'loadmodule':'u2r', 'multihop':'r2l',
                       'neptune':'dos', 'nmap':'probe', 'perl':'u2r', 'phf':'r2l',
                       'pod':'dos', 'portsweep':'probe', 'rootkit':'u2r',
                       'satan':'probe', 'smurf':'dos', 'spy':'r2l', 'teardrop':'dos',
                       'warezclient':'r2l', 'warezmaster':'r2l'}
    df_kddcup['label'] = df_kddcup['label'].replace(conversion_dict)
    df_kddcup = df_kddcup.query("label != 'u2r'")
    df_y = pd.DataFrame(df_kddcup.label, columns=["label"], dtype="category")
    df_kddcup.drop(["label"], inplace=True, axis=1)
    x_kddcup = df_kddcup[NUMERIC_COLS].values
    x_kddcup = preprocessing.scale(x_kddcup)
    y_kddcup = df_y.label.cat.codes.to_numpy()
    return x_kddcup, y_kddcup

def create_tf_model(input_size, num_of_class):
    """ This method creates the tensorflow classification model """
    model_kddcup = tf.keras.Sequential([
        tf.keras.layers.Dense(200, input_dim=input_size, activation=tf.nn.relu),
        tf.keras.layers.Dense(500, activation=tf.nn.relu),
        tf.keras.layers.Dense(200, activation=tf.nn.relu),
        tf.keras.layers.Dense(num_of_class),
        tf.keras.layers.Activation(tf.nn.softmax)
        ])
    model_kddcup.compile(loss='categorical_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])
    return model_kddcup


if __name__ == '__main__':
    main()
