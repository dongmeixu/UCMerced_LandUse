import datetime

from keras import layers, Input
from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K
import os

from keras.engine import get_source_inputs
from keras.layers import Flatten, Dense, AveragePooling2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, \
    GlobalAveragePooling2D, GlobalMaxPooling2D, warnings
from keras.models import Model
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint
from override_image import ImageDataGenerator
import numpy as np
import matplotlib
from keras.utils import layer_utils

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 指定GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# 定义超参数
learning_rate = 0.0001
img_width = 256
img_height = 256
# nbr_train_samples = 1672
# nbr_validation_samples = 419
nbr_train_samples = 191
nbr_validation_samples = 51
nbr_epochs = 1000
batch_size = 32
img_channel = 3
# n_classes = 21
n_classes = 4

base_dir = '/media/files/xdm/classification/'
# model_dir = base_dir + 'weights/UCMerced_LandUse/'
model_dir = base_dir + 'weights/2015_4_classes/'


# 定义训练集以及验证集的路径
# train_data_dir = base_dir + 'data/UCMerced_LandUse/train_split'
# val_data_dir = base_dir + 'data/UCMerced_LandUse/val_split'
train_data_dir = base_dir + 'data/2015_4_classes/aug_256/train_split'
val_data_dir = base_dir + 'data/2015_4_classes/aug_256/val_split'

# # 共21类(影像中所有地物的名称)
# ObjectNames = ['agricultural', 'airplane', 'baseballdiamond', 'beach',
#                'buildings', 'chaparral', 'denseresidential', 'forest',
#                'freeway', 'golfcourse', 'harbor', 'intersection',
#                'mediumresidential', 'mobilehomepark', 'overpass', 'parkinglot',
#                'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt'
#                ]

# 共21类(影像中所有地物的名称)
ObjectNames = ['building', 'other', 'water', 'zhibei']


WEIGHTS_PATH = '/media/files/xdm/classification/pre_weights/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = '/media/files/xdm/classification/pre_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(
        64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = WEIGHTS_PATH
        else:
            weights_path = WEIGHTS_PATH_NO_TOP

        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

        if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
            warnings.warn('You are using the TensorFlow backend, yet you '
                          'are using the Theano '
                          'image data format convention '
                          '(`image_data_format="channels_first"`). '
                          'For best performance, set '
                          '`image_data_format="channels_last"` in '
                          'your Keras config '
                          'at ~/.keras/keras.json.')
    elif weights is not None:
        model.load_weights(weights)

    return model


if __name__ == '__main__':
    print('Loading ResNet50 Weights ...')
    ResNet50_notop = ResNet50(include_top=False, weights='imagenet',
                              input_tensor=None, input_shape=(img_width, img_height, img_channel))
    ResNet50_notop.summary()

    print('Adding Average Pooling Layer and Softmax Output Layer ...')
    output = ResNet50_notop.get_layer(index=-1).output  # Shape: (6, 6, 2048)
    # output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
    output = Flatten(name='flatten')(output)
    output = Dense(n_classes, activation='softmax', name='predictions')(output)

    ResNet50_model = Model(ResNet50_notop.input, output)
    ResNet50_model.summary()

    optimizer = SGD(lr=learning_rate, momentum=0.9, decay=0.001, nesterov=True)
    ResNet50_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # autosave best Model
    # best_model_file = model_dir + "ResNet50_UCM_weights.h5"
    best_model_file = model_dir + "ResNet50_2015_4classes_weights.h5"
    best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose=1, save_best_only=True)

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        samplewise_center=True,  # 输入数据集去中心化，按feature执行
        rescale=1. / 255,  # 重缩放因子
        shear_range=0.1,  # 剪切强度（逆时针方向的剪切变换角度）
        zoom_range=0.1,  # 随机缩放的幅度
        rotation_range=10.,  # 图片随机转动的角度
        width_shift_range=0.1,  # 图片水平偏移的幅度
        height_shift_range=0.1,  # 图片竖直偏移的幅度
        horizontal_flip=True,  # 进行随机水平翻转
        vertical_flip=True,  # 进行随机竖直翻转
    )

    # this is the augmentation configuration we will use for validation:
    # only rescaling
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=True,
        # save_to_dir = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM/data/visualization',
        # save_prefix = 'aug',
        classes=ObjectNames,
        class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=True,
        # save_to_dir = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM/data/visulization',
        # save_prefix = 'aug',
        classes=ObjectNames,
        class_mode='categorical')

    begin = datetime.datetime.now()
    print('[{}] Creating and compiling model...'.format(str(datetime.datetime.now())))

    # Model visualization
    from keras.utils.vis_utils import plot_model

    plot_model(ResNet50_model, to_file=model_dir + 'ResNet50_2015_4classes_{}_{}'.format(batch_size, nbr_epochs),
               show_shapes=True)

    H = ResNet50_model.fit_generator(
        train_generator,
        samples_per_epoch=nbr_train_samples,
        nb_epoch=nbr_epochs,
        validation_data=validation_generator,
        nb_val_samples=nbr_validation_samples,
        callbacks=[best_model])

    # plot the training loss and accuracy
    plt.figure()
    N = nbr_epochs
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")

    plt.title("Training Loss and Accuracy on Satellite")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    # 存储图像，注意，必须在show之前savefig，否则存储的图片一片空白
    plt.savefig(model_dir + "ResNet50_2015_4classes_{}_{}.png".format(batch_size, nbr_epochs))
    # plt.show()

    print('[{}]Finishing training...'.format(str(datetime.datetime.now())))

    end = datetime.datetime.now()
    print("总的训练时间为：", end - begin)
