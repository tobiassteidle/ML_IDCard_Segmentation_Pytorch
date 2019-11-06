import os
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import models

NO_OF_TRAINING_IMAGES = len(os.listdir('dataset/train/train_frames/image'))
NO_OF_VAL_IMAGES = len(os.listdir('dataset/train/val_frames/image'))

NO_OF_EPOCHS = 100
BATCH_SIZE = 8

IMAGE_SIZE = (256, 256)

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def main():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_image_generator = train_datagen.flow_from_directory('./dataset/train/train_frames',
                                                              target_size=IMAGE_SIZE, class_mode=None, batch_size=BATCH_SIZE)

    train_mask_generator = train_datagen.flow_from_directory('dataset/train/train_masks',
                                                             target_size=IMAGE_SIZE, class_mode=None, batch_size=BATCH_SIZE)

    val_image_generator = val_datagen.flow_from_directory('dataset/train/val_frames',
                                                          target_size=IMAGE_SIZE, class_mode=None, batch_size=BATCH_SIZE)

    val_mask_generator = val_datagen.flow_from_directory('dataset/train/val_masks',
                                                         target_size=IMAGE_SIZE, class_mode=None, batch_size=BATCH_SIZE)

    train_generator = zip(train_image_generator, train_mask_generator)
    val_generator = zip(val_image_generator, val_mask_generator)

    model = models.UUNET(input_size=(256, 256, 3))

    optimizer = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss=dice_coef_loss, optimizer=optimizer, metrics=[dice_coef])

    checkpoint = ModelCheckpoint('weights/', monitor=[dice_coef], verbose=1, save_best_only=True, mode='max')
    tensorboard = TensorBoard(log_dir='./logs')
    earlystopping = EarlyStopping(monitor=dice_coef, verbose=1, min_delta=0.01, patience=3, mode='max')

    callbacks_list = [checkpoint, tensorboard, earlystopping]

    results = model.fit_generator(train_generator, epochs=NO_OF_EPOCHS,
                              steps_per_epoch=(NO_OF_TRAINING_IMAGES // BATCH_SIZE),
                              validation_data=val_generator,
                              validation_steps=(NO_OF_VAL_IMAGES // BATCH_SIZE),
                              callbacks=callbacks_list)
    model.save('Model.h5')


if __name__ == '__main__':
    main()
