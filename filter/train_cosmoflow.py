import chainer
import chainermn
from chainer import datasets, training
from chainer.training import extensions
import argparse

# Local imports
from models.cosmoflow import CosmoFlow
from utils.cosmoflow_data_prep import CosmoDataset

import matplotlib

matplotlib.use('Agg')


def main():

    #  Create ChainerMN communicator.

    comm = chainermn.create_communicator('hierarchical')
    device = comm.intra_rank
    chainer.backends.cuda.get_device_from_id(device).use()

    # Input data and label
    training_data = CosmoDataset("/groups2/gaa50004/cosmoflow_data")
    # training_data = temp_data_prep()  # temp data
    print("Fetching data successful")
    print("Found %d training samples" % training_data.__len__())
    train, val = datasets.split_dataset_random(
        training_data, first_size=(int(training_data.__len__() * 0.80)))

    if comm.rank != 0:
        train = chainermn.datasets.create_empty_dataset(train)
        val = chainermn.datasets.create_empty_dataset(val)

    train_iterator = chainermn.iterators.create_multi_node_iterator(
        chainer.iterators.SerialIterator(train, 1), comm)
    vali_iterator = chainermn.iterators.create_multi_node_iterator(
        chainer.iterators.SerialIterator(val, 1, repeat=False, shuffle=False), comm)

    model = CosmoFlow(comm)

    model.to_gpu()  # Copy the model to the GPU

    optimizer = chainer.optimizers.Adam()

    optimizer.setup(model)
    # Create the updater, using the optimizer
    updater = training.StandardUpdater(train_iterator, optimizer, device=device)

    # Set up a trainer
    trainer = training.Trainer(updater, (10, 'epoch'), out='result')
    trainer.extend(extensions.Evaluator(vali_iterator, model, device=device))

    log_interval = (1, 'epoch')
    if comm.rank == 0:
        trainer.extend(extensions.DumpGraph('main/loss'))
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.PrintReport(['epoch' 'Validation loss', 'lr']), trigger=log_interval)
        trainer.extend(extensions.ProgressBar(update_interval=10))
        print("Starting Training ")
    trainer.run()


if __name__ == "__main__":

    main()


