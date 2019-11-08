import numpy as np
import os
import random as rn
import tensorflow as tf
import time
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import models
from utils import metrics

NO_OF_TRAINING_IMAGES = len(os.listdir('dataset/train/train_frames/image'))
NO_OF_VAL_IMAGES = len(os.listdir('dataset/train/val_frames/image'))

NO_OF_EPOCHS = 100
BATCH_SIZE = 8

IMAGE_SIZE = (256, 256)

SEED = 230
rn.seed(SEED)
np.random.seed(SEED)
tf.set_random_seed(SEED)


def main():
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    train_image_generator = train_datagen.flow_from_directory('./dataset/train/train_frames',
                                                              target_size=IMAGE_SIZE,
                                                              class_mode=None,
                                                              batch_size=BATCH_SIZE,
                                                              color_mode='grayscale',
                                                              seed=SEED)

    train_mask_generator = train_datagen.flow_from_directory('dataset/train/train_masks',
                                                             target_size=IMAGE_SIZE,
                                                             class_mode=None,
                                                             batch_size=BATCH_SIZE,
                                                             color_mode='grayscale',
                                                             seed=SEED)

    val_datagen = ImageDataGenerator(rescale=1. / 255)
    val_image_generator = val_datagen.flow_from_directory('dataset/train/val_frames',
                                                          target_size=IMAGE_SIZE,
                                                          class_mode=None,
                                                          batch_size=BATCH_SIZE,
                                                          color_mode='grayscale',
                                                          seed=SEED)

    val_mask_generator = val_datagen.flow_from_directory('dataset/train/val_masks',
                                                         target_size=IMAGE_SIZE,
                                                         class_mode=None,
                                                         batch_size=BATCH_SIZE,
                                                         color_mode='grayscale',
                                                         seed=SEED)

    train_generator = zip(train_image_generator, train_mask_generator)
    val_generator = zip(val_image_generator, val_mask_generator)

    # build model
    model = models.UNET(input_size=(256, 256, 1))
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy', metrics.mean_iou])

    # configure callbacks
    '''
    checkpoint = ModelCheckpoint("model.h5", verbose=1, save_best_only=True, save_weights_only=False,
                                 monitor='val_mean_iou', mode='max')
    earlystopping = EarlyStopping(patience=10, verbose=1, monitor='val_mean_iou', mode='max')
    '''

    checkpoint = ModelCheckpoint("model.h5", verbose=1, save_best_only=True, save_weights_only=False)
    earlystopping = EarlyStopping(patience=10, verbose=1)

    reduce_lr = ReduceLROnPlateau(factor=0.2,
                                  patience=3,
                                  verbose=1,
                                  min_delta=0.000001)
    tensorboard = TensorBoard(log_dir='./logs/' + time.strftime("%Y%m%d_%H%M%S"), histogram_freq=0,
                              write_graph=True, write_images=True)

    # train model
    model.fit_generator(train_generator, epochs=NO_OF_EPOCHS,
                        steps_per_epoch=(NO_OF_TRAINING_IMAGES // BATCH_SIZE),
                        validation_data=val_generator,
                        validation_steps=(NO_OF_VAL_IMAGES // BATCH_SIZE),
                        callbacks=[checkpoint, earlystopping, reduce_lr, tensorboard])


if __name__ == '__main__':
    main()
