class my_callbacks(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logd = {}, desire_loss):

        if (logs.get('loss') < desire_loss):
            self.model.stop_training = True

