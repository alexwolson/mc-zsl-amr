from keras.utils import plot_model
from do_word_encodings import load_data
from keras.layers import Input
from devise import build_model

image_features = Input(shape=(1000,), name="image_feature_input")
caption_features = Input(shape=(88,), name="caption_feature_input")
(feature_data_matrix, image_scores, data, embedding_matrix, label_vectors) = load_data(20000, 100)
model = build_model(image_features, caption_features, embedding_matrix)
plot_model(model, to_file='model.png')