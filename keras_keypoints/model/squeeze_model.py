from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda, concatenate, MaxPool2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D

from keras.layers.merge import Multiply
from keras.regularizers import l2
from keras.initializers import random_normal,constant, TruncatedNormal


def _fire_layer(name, input, s1x1, e1x1, e3x3, stdd=0.01, WEIGHT_DECAY=0.001):
    """
    wrapper for fire layer constructions
    :param name: name for layer
    :param input: previous layer
    :param s1x1: number of filters for squeezing
    :param e1x1: number of filter for expand 1x1
    :param e3x3: number of filter for expand 3x3
    :param stdd: standard deviation used for intialization
    :return: a keras fire layer
    """

    sq1x1 = Conv2D(
        name=name + '/squeeze1x1', filters=s1x1, kernel_size=(1, 1), strides=(1, 1), use_bias=True,
        padding='SAME', kernel_initializer=TruncatedNormal(stddev=stdd), activation="relu",
        kernel_regularizer=l2(WEIGHT_DECAY))(input)

    ex1x1 = Conv2D(
        name=name + '/expand1x1', filters=e1x1, kernel_size=(1, 1), strides=(1, 1), use_bias=True,
        padding='SAME', kernel_initializer=TruncatedNormal(stddev=stdd), activation="relu",
        kernel_regularizer=l2(WEIGHT_DECAY))(sq1x1)

    ex3x3 = Conv2D(
        name=name + '/expand3x3', filters=e3x3, kernel_size=(3, 3), strides=(1, 1), use_bias=True,
        padding='SAME', kernel_initializer=TruncatedNormal(stddev=stdd), activation="relu",
        kernel_regularizer=l2(WEIGHT_DECAY))(sq1x1)

    return concatenate([ex1x1, ex3x3], axis=3)

def squeezenet(x,WEIGHT_DECAY):


    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME", activation='relu',
                   use_bias=True, kernel_initializer=TruncatedNormal(stddev=0.001),
                   kernel_regularizer=l2(WEIGHT_DECAY))(x)

    pool1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='SAME', name="pool1")(conv1)

    fire2 = _fire_layer(name="fire2", input=pool1, s1x1=16, e1x1=64, e3x3=64)

    fire3 = _fire_layer(
        'fire3', fire2, s1x1=16, e1x1=64, e3x3=64)
    pool3 = MaxPool2D(
        pool_size=(3, 3), strides=(2, 2), padding='SAME', name='pool3')(fire3)

    fire4 = _fire_layer(
        'fire4', pool3, s1x1=32, e1x1=128, e3x3=128)
    fire5 = _fire_layer(
        'fire5', fire4, s1x1=32, e1x1=128, e3x3=128)

    pool5 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='SAME', name="pool5")(fire5)

    fire6 = _fire_layer(
        'fire6', fire5, s1x1=48, e1x1=192, e3x3=192)
    fire7 = _fire_layer(
        'fire7', fire6, s1x1=48, e1x1=192, e3x3=192)
    fire8 = _fire_layer(
        'fire8', fire7, s1x1=64, e1x1=256, e3x3=256)
    fire9 = _fire_layer(
        'fire9', fire8, s1x1=64, e1x1=256, e3x3=256)

    # pool9 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='SAME', name="pool9")(fire9)

    return fire9


def relu(x): return Activation('relu')(x)


def conv(x, nf, ks, name, weight_decay):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[1]) if weight_decay else None

    x = Conv2D(nf, (ks, ks), padding='same', name=name,
               kernel_regularizer=kernel_reg,
               bias_regularizer=bias_reg,
               kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(x)
    return x


def stage1_block(x, num_p, branch, weight_decay):
    # Block 1
    x = conv(x, 128, 3, "Mconv1_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "Mconv2_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "Mconv3_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 1, "Mconv4_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv5_stage1_L%d" % branch, (weight_decay, 0))

    return x


def stageT_block(x, num_p, stage, branch, weight_decay):
    # Block 1
    x = conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch), (weight_decay, 0))

    return x


def apply_mask(x, mask1, mask2, num_p, stage, branch):
    w_name = "weight_stage%d_L%d" % (stage, branch)
    if num_p == 38:
        w = Multiply(name=w_name)([x, mask1]) # vec_weight

    else:
        w = Multiply(name=w_name)([x, mask2])  # vec_heat
    return w


def get_training_model(weight_decay):

    stages = 6
    np_branch1 = 38
    np_branch2 = 19

    img_input_shape = (None, None, 3)
    vec_input_shape = (None, None, 38)
    heat_input_shape = (None, None, 19)

    inputs = []
    outputs = []

    img_input = Input(shape=img_input_shape)
    vec_weight_input = Input(shape=vec_input_shape)
    heat_weight_input = Input(shape=heat_input_shape)

    inputs.append(img_input)
    inputs.append(vec_weight_input)
    inputs.append(heat_weight_input)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]

    # VGG
    stage0_out = squeezenet(img_normalized, weight_decay)

    # stage 1 - branch 1 (PAF)
    stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, weight_decay)
    w1 = apply_mask(stage1_branch1_out, vec_weight_input, heat_weight_input, np_branch1, 1, 1)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay)
    w2 = apply_mask(stage1_branch2_out, vec_weight_input, heat_weight_input, np_branch2, 1, 2)

    x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

    outputs.append(w1)
    outputs.append(w2)

    # stage sn >= 2
    for sn in range(2, stages + 1):
        # stage SN - branch 1 (PAF)
        stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, weight_decay)
        w1 = apply_mask(stageT_branch1_out, vec_weight_input, heat_weight_input, np_branch1, sn, 1)

        # stage SN - branch 2 (confidence maps)
        stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, weight_decay)
        w2 = apply_mask(stageT_branch2_out, vec_weight_input, heat_weight_input, np_branch2, sn, 2)

        outputs.append(w1)
        outputs.append(w2)

        if (sn < stages):
            x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

    model = Model(inputs=inputs, outputs=outputs)

    return model


def get_testing_model():
    stages = 6
    np_branch1 = 38
    np_branch2 = 19

    img_input_shape = (None, None, 3)

    img_input = Input(shape=img_input_shape)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]

    # VGG
    stage0_out = squeezenet(img_normalized, None)

    # stage 1 - branch 1 (PAF)
    stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, None)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, None)

    x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

    # stage t >= 2
    stageT_branch1_out = None
    stageT_branch2_out = None
    for sn in range(2, stages + 1):
        stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, None)
        stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, None)

        if (sn < stages):
            x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

    model = Model(inputs=[img_input], outputs=[stageT_branch1_out, stageT_branch2_out])

    return model