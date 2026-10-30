from tensorflow.keras.utils import Sequence
from keras.callbacks import Callback


class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler implementing linear warmup and cosine decay."""

    def __init__(self, warmup_steps, total_steps, target_lr=0.001, start_lr=0.0, hold=0):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.target_lr = target_lr
        self.start_lr = start_lr
        self.hold = hold
        self.global_step = 0  # Initialize step counter

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "learning_rate"):
            raise ValueError('Optimizer must have a "learning_rate" attribute.')

        # Compute new learning rate
        new_lr = self.lr_warmup_cosine_decay(self.global_step)
        self.model.optimizer.learning_rate.assign(new_lr)  # Assign new LR

        print(f"\nEpoch {epoch}: Learning rate is {float(new_lr)}")
        self.global_step += 1  # Increment step counter

    def lr_warmup_cosine_decay(self, global_step):
        """Computes learning rate with linear warmup and cosine decay."""
        if global_step >= self.total_steps:
            return self.target_lr  # Ensure no overshooting beyond total steps

        # Cosine decay
        learning_rate = 0.5 * self.target_lr * (1 + np.cos(
            np.pi * (global_step - self.warmup_steps - self.hold) / float(
                self.total_steps - self.warmup_steps - self.hold)))

        # Linear warmup
        warmup_lr = self.target_lr * (global_step / self.warmup_steps)

        # Apply warmup, hold, and decay logic
        if self.hold > 0:
            learning_rate = np.where(global_step > self.warmup_steps + self.hold, learning_rate, self.target_lr)

        learning_rate = np.where(global_step < self.warmup_steps, warmup_lr, learning_rate)
        return learning_rate


class ClearMemory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        K.clear_session()
        gc.collect()

class DataGenerator(Sequence):
    def __init__(self, X_dyn, X_stat, y, batch_size, shuffle=True):
        self.X_dyn = X_dyn
        self.X_stat = X_stat
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(y))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.y) / self.batch_size))

    def __getitem__(self, index):
        # Generate batch indices
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X_dyn_batch = self.X_dyn[batch_indices]
        X_stat_batch = self.X_stat[batch_indices]
        y_batch = self.y[batch_indices]

        return (X_dyn_batch, X_stat_batch), y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class BatchProgressCallback(Callback):
    def on_train_batch_end(self, batch, logs=None):
        print(f"Batch {batch + 1} processed. Loss: {logs.get('loss'):.4f}")

    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nStarting Epoch {epoch + 1}...\n")
