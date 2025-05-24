from function import *
from sklearn.model_selection import train_test_split
import tensorflow 
#from tensorflow import keras
#from keras import to_categorical
import keras
from keras import Sequential
#from keras.layers import LSTM, Dense
#from keras.callbacks import TensorBoard
#from tensorflow import keras
#from keras.callbacks import TensorBoard
#from tensorflow import keras
label_map = {label:num for num, label in enumerate(actions)}
# print(label_map)
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)),allow_pickle=True)
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = keras.utils.to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir)
model = Sequential()
model.add(keras.layers.LSTM(64, return_sequences=True, activation='relu', input_shape=(30,63)))
model.add(keras.layers.LSTM(128, return_sequences=True, activation='relu'))
model.add(keras.layers.LSTM(64, return_sequences=False, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(actions.shape[0], activation='softmax'))
res = [.7, 0.2, 0.1]

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])
model.summary()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')