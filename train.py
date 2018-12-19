from keras import Model
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.initializers import RandomNormal
from keras.layers import Flatten, Dropout, Dense


def generate_model(application, num_class, img_size, pre_weights=None):
    if application == 'InceptionV3':
        base_model = InceptionV3(input_shape=(img_size, img_size, 3),
                                 include_top=False,
                                 weights=pre_weights)
    elif application == 'MobileNet':
        base_model = MobileNet(input_shape=(img_size, img_size, 3),
                               include_top=False,
                               weights=pre_weights)
    elif application == 'VGG19':
        base_model = VGG19(input_shape=(img_size, img_size, 3),
                           weights=pre_weights,
                           include_top=None)
    elif application == 'InceptionResNetV2':
        base_model = InceptionResNetV2(input_shape=(img_size, img_size, 3),
                                       weights=pre_weights,
                                       include_top=None)
    elif application == 'Xception':
        base_model = Xception(input_shape=(img_size, img_size, 3),
                              weights=pre_weights,
                              include_top=None)
    else:
        raise Exception('No specific aplication type!')

    x = base_model.output
    feature = Flatten(name='feature')(x)
    predictions = Dropout(0.5)(feature)
    # x = GlobalAveragePooling2D()(x)
    # predictions = Dense(1024, activation='relu')(x)
    predictions = Dense(num_class, activation='softmax',
                        name='pred',
                        kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(predictions)
    model = Model(inputs=base_model.input, outputs=[predictions, feature])
    Model.summary(model)
    print(model.output)
    return model


generate_model("Xception", 9, 224, pre_weights=None)
