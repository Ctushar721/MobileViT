import keras
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import pandas as pd


def test_model():
    # Load the model
    model = keras.models.load_model('mobilevit2.h5')

    df = pd.read_csv('./x_rays/test_Sheet1.csv')

    datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = datagen.flow_from_dataframe(df,
                                               directory='./x_rays/test_images',
                                               x_col='Image Index',
                                               y_col='Finding Labels',
                                               target_size=(64, 64),
                                               batch_size=32,
                                               class_mode='categorical')
    results = model.evaluate(training_set, batch_size=32)
    print(results)
    predictions = model.predict(training_set)
    print(predictions)


test_model()