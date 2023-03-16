import jax
import flax
import tensorflow as tf
import tensorflow_datasets as tfds
from clu import deterministic_data


class Dataset:
    def __init__(self, dataset_name="mnist", batch_size=128):
        self.dataset_name = dataset_name
        self.dataset_builder = tfds.builder(dataset_name)
        self.dataset_builder.download_and_prepare()
        self.train_split = tfds.split_for_jax_process("train", drop_remainder=True)
        self.batch_size = batch_size

    def _preprocess_mnist(self, batch):
        images = tf.cast(batch["image"], "float32") / 255
        images = images * 2 - 1
        labels = batch["label"]
        return images, labels

    def _preprocess_cifar10(self, batch):
        images = tf.cast(batch["image"], "float32") / 255
        images = images * 2 - 1
        labels = batch["label"]
        return images, labels

    def create_dataset(self):
        preprocess_fn = self._preprocess_mnist if self.dataset_name == "mnist" else self._preprocess_cifar10

        train_ds = deterministic_data.create_dataset(self.dataset_builder, split=self.train_split, rng=jax.random.PRNGKey(0), shuffle_buffer_size=100, batch_dims=[jax.local_device_count(), self.batch_size // jax.device_count()], num_epochs=None, preprocess_fn=preprocess_fn, shuffle=True)
        return train_ds

    def create_input_iter(self, ds):
        def _prepare(xs):
            def _f(x):
                x = x._numpy()
                return x

            return jax.tree_util.tree_map(_f, xs)

        it = map(_prepare, ds)
        it = flax.jax_utils.prefetch_to_device(it, 2)
        return it


# MNIST
mnist_dataset = Dataset("mnist")
mnist_train_ds = mnist_dataset.create_dataset()
mnist_batches = mnist_dataset.create_input_iter(mnist_train_ds)

# CIFAR-10
cifar10_dataset = Dataset("cifar10")
cifar10_train_ds = cifar10_dataset.create_dataset()
cifar10_batches = cifar10_dataset.create_input_iter(cifar10_train_ds)
