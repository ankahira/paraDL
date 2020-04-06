import chainer as ch
import chainermn
from chainer import datasets, training
from chainer.training import extensions
import argparse
import cupy as cp
import chainermnx
import chainer
from datetime import datetime


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
    chainer.disable_experimental_feature_warning = True

    parser = argparse.ArgumentParser(description='CosmoFlow Multi-Node Training')
    parser.add_argument('--batchsize', '-B', type=int, default=32, help='Learning minibatch size')
    parser.add_argument('--epochs', '-E', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--out', '-o', default='results', help='Output directory')
    args = parser.parse_args()

    batch_size = args.batchsize
    epochs = args.epochs
    out = args.out
    #  Create ChainerMN communicator.
    comm = chainermnx.create_communicator("spatial_nccl")
    device = comm.intra_rank

    # Input data and label
    train = CosmoDataset("/groups2/gaa50004/cosmoflow_data")

    if comm.rank != 0:
        train = chainermn.datasets.create_empty_dataset(train)
        # test = chainermn.datasets.create_empty_dataset(test)

    train_iterator = chainermn.iterators.create_multi_node_iterator(
        chainer.iterators.MultithreadIterator(train, batch_size, n_threads=20, shuffle=True), comm)
    # vali_iterator = chainermn.iterators.create_multi_node_iterator(
    #     chainer.iterators.MultithreadIterator(test, batch_size, repeat=False, shuffle=False, n_threads=20),
    #     comm)
    # train_iterator = ch.iterators.SerialIterator(train, batch_size, shuffle=True)
    # vali_iterator = ch.iterators.SerialIterator(test, batch_size, repeat=False, shuffle=False)
    model = CosmoFlow(comm)

    # print("Model Created successfully")
    ch.backends.cuda.get_device_from_id(device).use()
    model.to_gpu()  # Copy the model to the GPU

    optimizer = ch.optimizers.Adam()

    optimizer.setup(model)
    # Create the updater, using the optimizer
    updater = training.StandardUpdater(train_iterator, optimizer, device=device)

    # Set up a trainer
    trainer = training.Trainer(updater, (epochs, 'epoch'), out=out)
    # trainer.extend(extensions.Evaluator(vali_iterator, model, device=device))

    log_interval = (1, 'epoch')
    filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".log"

    if comm.rank == 0:
        trainer.extend(extensions.DumpGraph('main/loss'))
        trainer.extend(extensions.LogReport(trigger=log_interval, filename=filename))
        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.PrintReport(['epoch' 'Validation loss', 'lr']), trigger=log_interval)
        trainer.extend(extensions.ProgressBar(update_interval=1))
        print("Starting Training ")
    trainer.run()


if __name__ == "__main__":

    main()


