import tensorflow
#import tensorflow.python.platform.build_info as build
import keras
from tensorflow.python.platform import build_info as build

print(build.build_info['cuda_version'])

print()

print(build.build_info)

#print(tf.test.is_built_with_cuda())
print()

print(build.build_info['cuda_version'])

print()

print(build.build_info['cudnn_version'])
