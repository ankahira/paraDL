import numpy as np
import chainer as ch
from chainer import datasets
import chainer.functions as F
from chainer.dataset import concat_examples
from chainer.backends.cuda import to_cpu


# Local imports
from models.CosmoNet import CosmoNet

import matplotlib

matplotlib.use('Agg')


def data_prep():
    X = np.random.rand(40, 1, 64, 64, 64).astype(np.float32)

    Y = np.random.rand(40, 2).astype(np.float32)

    train, test = datasets.split_dataset_random(datasets.TupleDataset(X, Y), first_size=30)

    train_iterator = ch.iterators.SerialIterator(train, 10)
    test_iter = ch.iterators.SerialIterator(test, 10, repeat=False, shuffle=False)

    return train_iterator, test_iter


def train(model, train_iter, test_iter, optimizer, gpu_id, epochs):

    while train_iter.epoch < epochs:

        # ---------- One iteration of the training loop ----------
        train_batch = train_iter.next()
        image_train, target_train = concat_examples(train_batch, gpu_id)

        # Calculate the prediction of the network
        prediction_train = model(image_train)

        # Calculate the loss with softmax_cross_entropy
        loss = F.mean_squared_error(prediction_train, target_train)

        # Calculate the gradients in the network
        model.cleargrads()
        loss.backward()

        # Update all the trainable parameters
        optimizer.update()
        # --------------------- until here ---------------------

        # Check the validation accuracy of prediction after every epoch
        if train_iter.is_new_epoch:  # If this iteration is the final iteration of the current epoch

            # Display the training loss
            print('Epoch:{:02d} train_loss:{:.04f} '.format(train_iter.epoch, float(to_cpu(loss.array))), end='')

            test_losses = []
            test_accuracies = []
            while True:
                test_batch = test_iter.next()
                image_test, target_test = concat_examples(test_batch, gpu_id)

                # Forward the test data
                prediction_test = model(image_test)

                # Calculate the loss
                loss_test = F.mean_squared_error(prediction_test, target_test)
                test_losses.append(to_cpu(loss_test.array))

                # Calculate the R2
                accuracy = F.r2_score(prediction_test, target_test)
                accuracy.to_cpu()
                test_accuracies.append(accuracy.array)

                if test_iter.is_new_epoch:
                    test_iter.reset()
                    break

            print('val_loss:{:.04f} '.format(np.mean(test_losses)))


def main():

    # Input data and label

    train_iterator, test_iterator = data_prep()

    model = CosmoNet()

    gpu_id = -1  # Set to -1 if you use CPU
    if gpu_id >= 0:
        model.to_gpu(gpu_id)

    optimizer = ch.optimizers.Adam()

    optimizer.setup(model)

    train(model, train_iterator, test_iterator, optimizer, gpu_id, 10)


if __name__ == "__main__":

    main()


