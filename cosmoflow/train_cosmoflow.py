import chainer as ch
import chainermn
from chainer import datasets, training
from chainer.training import extensions
import argparse
import cupy as cp
import chainermnx


# Local imports
from models.spatial_cosmoflow import CosmoFlow
from utils.cosmoflow_data_prep import CosmoDataset

import matplotlib

matplotlib.use('Agg')


def main():
    # These two lines help with memory. If they are not included training runs out of memory.
    # Use them till you the real reason why its running out of memory

    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)

    parser = argparse.ArgumentParser(description='CosmoFlow Multi-Node Training')
    args = parser.parse_args()

    #  Create ChainerMN communicator.
    comm = chainermnx.create_communicator("spatial_nccl")
    device = comm.intra_rank

    # Input data and label
    training_data = CosmoDataset("/groups2/gaa50004/cosmoflow_data")
    # training_data = temp_data_prep()  # temp data
    # print("Fetching data successful")
    # print("Found %d training samples" % training_data.__len__())
    train, test = datasets.split_dataset_random(
        training_data, first_size=(int(training_data.__len__() * 0.80)))
    train_iterator = ch.iterators.SerialIterator(train, 1)
    vali_iterator = ch.iterators.SerialIterator(test, 1, repeat=False, shuffle=False)

    model = CosmoFlow(comm)

    # print("Model Created successfully")
    ch.backends.cuda.get_device_from_id(device).use()
    model.to_gpu()  # Copy the model to the GPU

    optimizer = ch.optimizers.Adam()

    optimizer.setup(model)
    # Create the updater, using the optimizer
    updater = training.StandardUpdater(train_iterator, optimizer, device=device)

    # Set up a trainer
    trainer = training.Trainer(updater, (100, 'iteration'), out='result')
    trainer.extend(extensions.Evaluator(vali_iterator, model, device=device))

    log_interval = (1, 'epoch')
    if comm.rank == 0:
        trainer.extend(extensions.DumpGraph('main/loss'))
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.PrintReport(['epoch' 'Validation loss', 'lr']), trigger=log_interval)
        trainer.extend(extensions.ProgressBar(update_interval=1))
        print("Starting Training ")
    trainer.run()


if __name__ == "__main__":

    main()


