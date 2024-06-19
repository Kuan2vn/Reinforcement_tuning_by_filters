import tensorflow as tf

# Check for available GPUs
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))

if physical_devices:
    try:
        # Set memory growth to avoid memory allocation errors
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_devices = tf.config.list_logical_devices('GPU')
        print("Logical GPUs:", logical_devices)
    except RuntimeError as e:
        print(e)

# Test TensorFlow with GPU
if physical_devices:
    print("GPU is available.")
    with tf.device('/GPU:0'):
        a = tf.random.normal((1000, 1000))
        b = tf.random.normal((1000, 1000))
        c = tf.matmul(a, b)
        print("Done with GPU computation")
else:
    print("GPU is not available.")
