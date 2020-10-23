from keras.layers import Lambda, Dense, BatchNormalization, Embedding, LSTM, concatenate, Input, Flatten, Dot, Activation
from keras.models import Model
from keras.losses import hinge, mean_squared_error
from keras.utils import plot_model
from keras import optimizers
from do_word_encodings import *
from keras import metrics
import pickle
import argparse


"""
to run:

python devise_medium.py -m VGG16 -e 1000 -v 30 -i n -db database_reduced.csv -d insert_comprehensive_description_of_all_param_changed -lr 0.001 -opt rmsprop -bs 128

"""
parser = argparse.ArgumentParser(description="Run the DeViSE model")
parser.add_argument("--imagemodel", '-m', default="VGG16")
parser.add_argument("--epochs", "-e", default=1000)
parser.add_argument("--val", "-v", default=20)
parser.add_argument("--image", "-i", default="n")
parser.add_argument("--database", "-db", default = "database_reduced.csv")
parser.add_argument("--description", "-d", default="random_exp")
parser.add_argument("--topk", "-t", default = 5)
parser.add_argument("--learning_rate", "-lr", default = 0.001)
parser.add_argument("--optimizer", "-opt", default ="rmsprop")
parser.add_argument("--batch_size", "-bs", default = 128)
parser.add_argument("--savemodel","-s",default="n")


args = parser.parse_args()

MARGIN = 0.2
INCORRECT_BATCH = 127
BATCH = int(args.batch_size)
IMAGE_DIM = 1000
WORD_DIM = 100
MAX_SEQUENCE_LENGTH = 26
MAX_NUM_WORDS = 20000
GLOVE_DIR = "glove"

#Taken from https://github.com/priyamtejaswin/devise-keras/tree/devise-rnn
def build_model(image_features, caption_features, embedding_matrix):

    #conv1_out = Conv1D(10, kernel_size=(2), strides=(1), padding='valid', dilation_rate=(1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(image_features)
    image_dense = Dense(WORD_DIM, name="image_dense")(image_features)
    image_output = BatchNormalization()(image_dense)
    selu1 = Activation('selu')(image_output)
    selu2 = Activation('selu')(selu1)
    selu3 = Activation('selu')(selu2)
    selu4 = Activation('selu')(selu3)
    selu5 = Activation('selu')(selu4)



    cap_embed = Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=WORD_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False,
        name="caption_embedding"
        )(caption_features)

    #flat = Flatten()(cap_embed)
    lstm_out = LSTM(100)(cap_embed)
    caption_output = Dense(WORD_DIM, name="lstm_dense")(lstm_out)
    caption_output = BatchNormalization()(lstm_out)
    output = Dot(axes=-1, normalize=True)([selu5, caption_output])
    #concated = concatenate([image_output, caption_output], axis=-1)

    if args.optimizer == 'rmsprop':
        opt = optimizers.rmsprop(lr=float(args.learning_rate))
    if args.optimizer == 'adam':
        opt = optimizers.adam(lr=float(args.learning_rate))
    if args.optimizer == 'adagrad':
        opt = optimizers.adagrad(lr=float(args.learning_rate))


    mymodel = Model(inputs=[image_features, caption_features], outputs=output)
    mymodel.compile(optimizer=opt, loss=mean_squared_error, metrics=['accuracy'])
    return mymodel

#Taken from https://github.com/priyamtejaswin/devise-keras/tree/devise-rnn
#Currently unused
def hinge_rank_loss(y_true, y_pred, TESTING=False):
    """
    Custom hinge loss per (image, label) example - Page4.

    Keras mandates the function signature to follow (y_true, y_pred)
    In devise:master model.py, this function accepts:
    - y_true as word_vectors
    - y_pred as image_vectors
    For the rnn_model, the image_vectors and the caption_vectors are concatenated.
    This is due to checks that Keras has enforced on (input,target) sizes
    and the inability to handle multiple outputs in a single loss function.
    These are the actual inputs to this function:
    - y_true is just a dummy placeholder of zeros (matching size check)
    - y_pred is concatenate([image_output, caption_output], axis=-1)
    The image, caption features are first separated and then used.
    """
    ## y_true will be zeros
    select_images = lambda x: x[:, :WORD_DIM]
    select_words = lambda x: x[:, WORD_DIM:]

    slice_first = lambda x: x[0:1 , :]
    slice_but_first = lambda x: x[1:, :]

    # separate the images from the captions==words
    image_vectors = Lambda(select_images, output_shape=(BATCH, WORD_DIM))(y_pred)
    word_vectors = Lambda(select_words, output_shape=(BATCH, WORD_DIM))(y_pred)

    # separate correct/wrong images
    correct_image = Lambda(slice_first, output_shape=(1, WORD_DIM))(image_vectors)
    wrong_images = Lambda(slice_but_first, output_shape=(INCORRECT_BATCH, WORD_DIM))(image_vectors)

    # separate correct/wrong words
    correct_word = Lambda(slice_first, output_shape=(1, WORD_DIM))(word_vectors)
    wrong_words = Lambda(slice_but_first, output_shape=(INCORRECT_BATCH, WORD_DIM))(word_vectors)

    # l2 norm
    l2 = lambda x: K.sqrt(K.sum(K.square(x), axis=1, keepdims=True))
    l2norm = lambda x: x/l2(x)

    # tiling to replicate correct_word and correct_image
    correct_words = K.tile(correct_word, (INCORRECT_BATCH,1))
    correct_images = K.tile(correct_image, (INCORRECT_BATCH,1))

    # converting to unit vectors
    correct_words = l2norm(correct_words)
    wrong_words = l2norm(wrong_words)
    correct_images = l2norm(correct_images)
    wrong_images = l2norm(wrong_images)

    # correct_image VS incorrect_words | Note the singular/plurals
    cost_images = MARGIN - K.sum(correct_images * correct_words, axis=1) + K.sum(correct_images * wrong_words, axis=1)
    cost_images = K.maximum(cost_images, 0.0)

    # correct_word VS incorrect_images | Note the singular/plurals
    cost_words = MARGIN - K.sum(correct_words * correct_images, axis=1) + K.sum(correct_words * wrong_images, axis=1)
    cost_words = K.maximum(cost_words, 0.0)

    # currently cost_words and cost_images are vectors - need to convert to scalar
    cost_images = K.sum(cost_images, axis=-1)
    cost_words  = K.sum(cost_words, axis=-1)

    if TESTING:
        # ipdb.set_trace()
        assert K.eval(wrong_words).shape[0] == INCORRECT_BATCH
        assert K.eval(correct_words).shape[0] == INCORRECT_BATCH
        assert K.eval(wrong_images).shape[0] == INCORRECT_BATCH
        assert K.eval(correct_images).shape[0] == INCORRECT_BATCH
        assert K.eval(correct_words).shape==K.eval(correct_images).shape
        assert K.eval(wrong_words).shape==K.eval(wrong_images).shape
        assert K.eval(correct_words).shape==K.eval(wrong_images).shape

    return (cost_words + cost_images) / INCORRECT_BATCH

if __name__ == "__main__":
    image_features = Input(shape=(IMAGE_DIM,), name="image_feature_input")
    caption_features = Input(shape=(MAX_SEQUENCE_LENGTH,), name="caption_feature_input")
    (imfeatures, ccembeddings, emmat, labels, class_labels) = load_data(MAX_NUM_WORDS, WORD_DIM, network=args.imagemodel, database = args.database)
    model = build_model(image_features, caption_features, emmat)
    if args.image == "y":
        plot_model(model, to_file='model.png')
    (imtrain, desctrain, labelstrain, imval, descval, labelsval) = zsl_split_data(ccembeddings, imfeatures, labels, class_labels, batch_size=BATCH, VALIDATION_SPLIT_PERCENTAGE=int(args.val)/100)
    model = train_model(model, imtrain, desctrain, labelstrain, imval, descval, labelsval, BATCH, int(args.epochs), args.description)
    print("args.savemodel: " + args.savemodel)
    if args.savemodel == "y":
        model.save("".join([c for c in args.description if c.isalnum()]) + ".h5")
