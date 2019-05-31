import extract_to_csv
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='\n')


def nn_model(input_length):
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[
            input_length]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])
    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model


def main():
    extract_to_csv.unzip_file('./datasets/italy_weather_ver2_0_12445_554866729.zip', './datasets/')
    # Read data from file
    w_df = pd.read_csv('./datasets/italy_weather_ver2_0_12445_554866729.csv', sep=';')
    # Clean data by replacing NaN to zero
    w_df_cl = w_df.fillna(0)
    # Show correlations
    print(w_df_cl.corr())
    # Keep only valuable data
    vd_df = w_df_cl.drop(['GRID_NO', 'LATITUDE', 'LONGITUDE', 'ALTITUDE', 'DAY', 'E0', 'ES0'], axis=1)
    # Split the data train-test
    train, test = train_test_split(vd_df, test_size=0.3)
    X_train = train.drop('ET0', axis=1)
    Y_train = train['ET0']

    # Build the model
    model = nn_model(len(X_train.keys()))
    EPOCHS = 11

    history = model.fit(
        X_train, Y_train,
        epochs=EPOCHS, validation_split=0.2, verbose=0,
        callbacks=[PrintDot()])

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())
    test_predictions = model.predict(test.drop('ET0', axis=1)).flatten()

    # plot predicted and true values
    plt.scatter(test_predictions, test['ET0'])
    plt.xlabel('Predictions [ET]')
    plt.ylabel('True Values [ET]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    plt.show()

    # evaluate the model
    scores = model.evaluate(X_train, Y_train)
    print(scores)
if __name__ == "__main__":
    main()