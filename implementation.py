import tensorflow as tf
import re


BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 100  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector
CLASS_SIZE = 2 #negative and positive
LSTM_UNITS = 64

stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than'}) #124 words

def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - stripping/adding punctuation
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    # -remove <br />
    # -make all lowercase
    # -remove stop words
    # -remove digits
    review_lower = review.lower().replace("<br />", " ")
    review_words = [word for word in re.split("\W+", review_lower) if word not in stop_words]
    processed_review = ' '.join(review_words)
    processed_review = ''.join([i for i in processed_review if not i.isdigit()])
    '''
    Eache processed_review is one string. It is suggested in the forum to split it to list of strings (each word one string)
    to increase the aquracy.
    '''
    return processed_review



def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """

    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=(), name="dropout_keep_prob")

    labels = tf.placeholder(tf.int32, [BATCH_SIZE, CLASS_SIZE], name="labels")
    input_data = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE], name="input_data")

    ######## Basic LSTM
    lstmCell = tf.contrib.rnn.BasicLSTMCell(LSTM_UNITS)  # Define lstmUnit
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=dropout_keep_prob)  # Regulization, avoid overfitting
    value, _ = tf.nn.dynamic_rnn(lstmCell, input_data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([LSTM_UNITS, CLASS_SIZE]))

    bias = tf.Variable(tf.constant(0.1, shape=[CLASS_SIZE]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)
    ######

    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32), name="accuracy")

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels), name="loss")
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss
