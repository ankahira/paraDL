import os
import chainer as ch
import chainermn
from chainer import datasets, training
from chainer.training import extensions
import argparse
import cupy as cp
import chainermnx
import chainer
from datetime import datetime
import chainer.functions as F
import chainer.links as L


# Local imports
from models.spatial_cosmoflow import CosmoFlow
from utils.cosmoflow_data_prep import CosmoDataset

import matplotlib

matplotlib.use('Agg')


def create_local_comm(comm):
    """Create a local communicator from the main communicator
    :arg: comm
    :return local comm
    """
    hs = comm.mpi_comm.allgather(os.uname()[1])
    host_list = []
    for h in hs:
        if h not in host_list:
            host_list.append(h)

    hosts = {k: v for v, k in enumerate(host_list)}

    local_comm = comm.split(hosts[os.uname()[1]], comm.intra_rank)
    return local_comm


def create_data_comm(comm):
    """Create a data communicator from the main communicator

    :arg: comm
    :return data comm
    """
    if comm.rank % 4 == 0:
        colour = 0
    else:
        colour = 1
    data_comm = comm.split(colour, comm.rank)
    return data_comm


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

    # Prepare communicators  communicator.
    comm = chainermnx.create_communicator("spatial_hybrid_nccl")
    local_comm = create_local_comm(comm)

    data_comm = create_data_comm(comm)
    device = comm.intra_rank

    if local_comm.rank == 0:
        if data_comm.rank == 0:
            train = CosmoDataset("/groups2/gaa50004/cosmoflow_data")
            # train, val = datasets.split_dataset_random(training_data, first_size=(int(training_data.__len__() * 0.80)))

        else:
            train = None
            #val = None
        train = chainermn.scatter_dataset(train, data_comm, shuffle=True)
        # val = chainermn.scatter_dataset(val, data_comm, shuffle=True)
    else:
        train = CosmoDataset("/groups2/gaa50004/cosmoflow_data")
        train = chainermn.datasets.create_empty_dataset(train)
        # val = chainermn.datasets.create_empty_dataset(val)

    train_iterator = chainermn.iterators.create_multi_node_iterator(
        chainer.iterators.MultithreadIterator(train, batch_size, n_threads=20, shuffle=True), local_comm)
    # vali_iterator = chainermn.iterators.create_multi_node_iterator(
    #     chainer.iterators.MultithreadIterator(val, batch_size, repeat=False, shuffle=False, n_threads=20),
    #     local_comm)

    model = CosmoFlow(local_comm)
    # model = L.Classifier(model, lossfun=F.mean_squared_error, accfun=F.mean_squared_error)

    # print("Model Created successfully")
    ch.backends.cuda.get_device_from_id(device).use()
    model.to_gpu()  # Copy the model to the GPU

    optimizer = chainermnx.create_hybrid_multi_node_optimizer_alpha(chainer.optimizers.Adam(), data_comm, local_comm)

    optimizer.setup(model)
    # Create the updater, using the optimizer
    updater = training.StandardUpdater(train_iterator, optimizer, device=device)

    # Set up a trainer
    trainer = training.Trainer(updater, (epochs, 'epoch'), out=out)
    # trainer.extend(extensions.Evaluator(vali_iterator, model, device=device))

    filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".log"
    log_interval = (1, 'epoch')
    if comm.rank == 0:
        trainer.extend(extensions.DumpGraph('main/loss'))
        trainer.extend(extensions.LogReport(trigger=log_interval, filename=filename))
        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
        trainer.extend(extensions.PlotReport(
            ['main/loss', 'validation/main/loss'], 'epoch', filename='loss.png'))
        trainer.extend(extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'], 'epoch', filename='accuracy.png'))

        trainer.extend(extensions.ProgressBar(update_interval=1))
        print("Starting Training ")

    trainer.run()


if __name__ == "__main__":

    main()


