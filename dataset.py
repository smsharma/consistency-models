import jax.numpy as jnp
import tensorflow_datasets as tfds


class DatasetIterator:
    def __init__(self, dataset_name, split, batch_size):
        if dataset_name == "mnist":
            self.ds = tfds.load("mnist", split=split, shuffle_files=True)
            self.preprocess_fn = self._preprocess_mnist
        elif dataset_name == "cifar10":
            self.ds = tfds.load("cifar10", split=split, shuffle_files=True)
            self.ds = self.ds.batch(batch_size)
            self.preprocess_fn = self._preprocess_cifar10
        else:
            raise ValueError("Unknown dataset: {}".format(dataset_name))

        self.ds = iter(self.ds)
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        batch = []
        for i in range(self.batch_size):
            try:
                item = next(self.ds)
            except StopIteration:
                self.ds = iter(self.ds)
                item = next(self.ds)
            batch.append(item)
        images, labels = self.preprocess_fn(batch)
        return images, labels

    def _preprocess_mnist(self, batch):
        mean, std = 0.1307, 0.3081
        images = [x["image"] for x in batch]
        # Flatten the images and normalize their pixel values to [0, 1]
        images = jnp.reshape(jnp.stack(images), (-1, 28 * 28)) / 255.0
        images = (images - mean) / std
        labels = [x["label"] for x in batch]
        # One-hot encode the labels
        labels = jnp.eye(10)[jnp.array(labels)]
        return images, labels

    def _preprocess_cifar10(self, batch):
        # Mean and standard deviation of the CIFAR-10 dataset
        mean = jnp.array([0.4914, 0.4822, 0.4465])
        std = jnp.array([0.2023, 0.1994, 0.2010])
        images = jnp.stack([x["image"] for x in batch])
        # Normalize the images using the mean and standard deviation
        images = (images - mean) / std
        labels = [x["label"] for x in batch]
        # One-hot encode the labels
        labels = jnp.eye(10)[jnp.array(labels)]
        return images, labels
