import tensorflow as tf

print("TensorFlow 버전:", tf.__version__)
print("CUDA 빌드:", tf.test.is_built_with_cuda())
print("GPU 장치:", tf.config.list_physical_devices('GPU'))

