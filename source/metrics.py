from sklearn.metrics import roc_auc_score
import tensorflow as tf



def roc_auc(y_true, y_pred):
    def try_roc_auc(y_true,y_pred):
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return 0
    return tf.py_function(try_roc_auc, (y_true, y_pred), tf.double)