from __future__ import print_function

import os
import numpy as np
from tqdm import tqdm
from keras.models import Model
from inception_resnet_v2 import InceptionResNetV2


WEIGHTS_DIR = './weights'
MODEL_DIR = './models'
OUTPUT_WEIGHT_FILENAME = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5'
OUTPUT_WEIGHT_FILENAME_NOTOP = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'


print('Instantiating an empty InceptionResNetV2 model...')
model = InceptionResNetV2(weights=None, input_shape=(299, 299, 3))

print('Loading weights from', WEIGHTS_DIR)
for layer in tqdm(model.layers):
    if layer.weights:
        weights = []
        for w in layer.weights:
            weight_name = os.path.basename(w.name).replace(':0', '')
            weight_file = layer.name + '_' + weight_name + '.npy'
            weight_arr = np.load(os.path.join(WEIGHTS_DIR, weight_file))

            # remove the "background class"
            if weight_file.startswith('Logits_bias'):
                weight_arr = weight_arr[1:]
            elif weight_file.startswith('Logits_kernel'):
                weight_arr = weight_arr[:, 1:]

            weights.append(weight_arr)
        layer.set_weights(weights)


print('Saving model weights...')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
model.save_weights(os.path.join(MODEL_DIR, OUTPUT_WEIGHT_FILENAME))

print('Saving model weights (no top)...')
model_notop = Model(model.inputs, model.get_layer('Conv2d_7b_1x1_Activation').output)
model_notop.save_weights(os.path.join(MODEL_DIR, OUTPUT_WEIGHT_FILENAME_NOTOP))
