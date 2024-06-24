import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices())

# If the 2nd print statement (tf.config) doesn't print an empty list, then the GPU is available for use. Awesome!