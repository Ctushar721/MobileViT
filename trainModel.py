import pandas as pd
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import MobileViT as mobVit
import math


def train_model():
    df = pd.read_csv('./x_rays/Sheet1.csv')
    entries = len(df)
    test_entries = math.floor(entries * 0.1)
    train_df = df[test_entries:]
    test_df = df[:test_entries]

    datagen = ImageDataGenerator(rescale=1. / 255)
    training_set = datagen.flow_from_dataframe(train_df,
                                               directory='./x_rays/images',
                                               x_col='Image Index',
                                               y_col='Finding Labels',
                                               target_size=(64, 64),
                                               batch_size=32,
                                               class_mode='categorical')

    num_channels = [16, 16, 24, 24, 24, 48, 64, 64, 80, 80, 96, 320]
    dim = [64, 80, 96]
    expansion_ratio = 2
    num_classes = 2
    input_shape = (64, 64, 3)
    model = mobVit.MobileViT(
        input_shape,
        num_channels,
        dim,
        expansion_ratio,
        num_classes=num_classes
    )
    model.summary()

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(training_set, epochs=10)
    model.save("mobilevit3.h5")
    print('training complete')


if __name__ == "__main__":
    train_model()