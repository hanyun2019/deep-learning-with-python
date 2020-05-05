# https://zhuanlan.zhihu.com/p/70232196

import numpy as np
import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Adding a dimension to the array -> new shape == (28, 28, 1)
train_images = train_images[..., None]
test_images = test_images[..., None]

# Getting the images in [0, 1] range.
train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

train_labels = train_labels.astype('int64')
test_labels = test_labels.astype('int64')

# dataset
train_ds = tf.data.Dataset.from_tensor_slices(
    (train_images, train_labels)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices(
    (test_images, test_labels)).batch(32)

# Model
class MyModel(tf.keras.Sequential):
    def __init__(self):
        super(MyModel, self).__init__([
          tf.keras.layers.Conv2D(32, 3, activation='relu'),
          tf.keras.layers.MaxPooling2D(),
          tf.keras.layers.Conv2D(64, 3, activation='relu'),
          tf.keras.layers.MaxPooling2D(),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(64, activation='relu'),
          tf.keras.layers.Dense(10, activation=None)
        ])

model = MyModel()

# optimizer
initial_learning_rate = 1e-4
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)

# checkpoint
checkpoint = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(checkpoint, './tf_ckpts', max_to_keep=3)

# loss function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# metric
train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# define a train step
@tf.function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_object(targets, predictions)
        loss += sum(model.losses)  # add other losses
    # compute gradients and update variables
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss_metric(loss)
    train_acc_metric(targets, predictions)

# define a test step
@tf.function
def test_step(inputs, targets):
    predictions = model(inputs, training=False)
    loss = loss_object(targets, predictions)
    test_loss_metric(loss)
    test_acc_metric(targets, predictions)

# train loop
epochs = 10
for epoch in range(epochs):
    print('Start of epoch %d' % (epoch,))
    # Iterate over the batches of the dataset
    for step, (inputs, targets) in enumerate(train_ds):
        train_step(inputs, targets)
        checkpoint.step.assign_add(1)
        # log every 20 step
        if step % 20 == 0:
            manager.save() # save checkpoint
            print('Epoch: {}, Step: {}, Train Loss: {}, Train Accuracy: {}'.format(
                epoch, step, train_loss_metric.result().numpy(),
                train_acc_metric.result().numpy())
            )
            train_loss_metric.reset_states()
            train_acc_metric.reset_states()

# do test
for inputs, targets in test_ds:
    test_step(inputs, targets)
print('Test Loss: {}, Test Accuracy: {}'.format(
    test_loss_metric.result().numpy(),
    test_acc_metric.result().numpy()))


# 麻雀虽小，但五脏俱全，这个实例包括数据加载，模型创建，以及模型训练和测试。
# 特别注意的是，这里我们将train和test的一个step通过tf.function转为Graph模式，可以加快训练速度，这是一种值得推荐的方式。
# 另外一点，上面的训练方式采用的是custom training loops，自由度较高;
# 另外一种训练方式是采用keras比较常规的compile和fit训练方式。