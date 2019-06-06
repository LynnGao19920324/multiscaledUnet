import keras.backend as K


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    #round:逐元素四舍五入；clip:逐元素将超出指定范围的数强制变为边界值
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    y_true, y_pred = K.round(y_true), K.round(y_pred)
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / (p + r + K.epsilon())

def acc(y_true, y_pred):
    y_true, y_pred = K.round(y_true), K.round(y_pred)
    return K.sum(K.cast(K.equal(y_true, y_pred), 'float32')) / K.cast(K.shape(y_true)[0], 'float32')
