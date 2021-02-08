import chainer as ch
import chainermn
from chainer import datasets, training
from chainer.training import extensions
import argparse
import cupy as cp
import chainermnx
import chainer
from datetime import datetime
import shutil
import os
import chainer.links as L
import chainer.functions as F



# Local imports
from models.spatial_cosmoflow import CosmoFlow
from utils.cosmoflow_data_prep import CosmoDataset
from utils.extras import temp_data_prep


import matplotlib

matplotlib.use('Agg')


def main():
    chainer.disable_experimental_feature_warning = True

    parser = argparse.ArgumentParser(description='CosmoFlow Multi-Node Training')
    parser.add_argument('--batchsize', '-B', type=int, default=32, help='Learning minibatch size')
    parser.add_argument('--epochs', '-E', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--out', '-o', default='results', help='Output directory')
    args = parser.parse_args()

    batch_size = args.batchsize
    epochs = args.epochs
    out = args.out

    # Clean up logs and directories from previous runs. This is temporary. In the future just add time stamps to logs

    # Directories are created later by the reporter.
    try:
        shutil.rmtree(out)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
    # Create new output dirs
    try:
        os.makedirs(out)
    except OSError:
        pass
    #  Create ChainerMN communicator.
    comm = chainermnx.create_communicator("spatial_nccl")
    device = comm.intra_rank

    # Input data and label
    train = CosmoDataset("/groups2/gaa50004/cosmoflow_data")

    if comm.rank != 0:
        train = chainermn.datasets.create_empty_dataset(train)
        # test = chainermn.datasets.create_empty_dataset(test)

    train_iterator = chainermnx.iterators.create_multi_node_iterator(
        chainer.iterators.MultithreadIterator(train, batch_size, n_threads=80,  shuffle=True), comm)

    # train_iterator = chainer.iterators.MultithreadIterator(train, batch_size, n_threads=80, shuffle=True)
    # vali_iterator = chainermn.iterators.create_multi_node_iterator(
    #     chainer.iterators.MultithreadIterator(test, batch_size, repeat=False, shuffle=False, n_threads=20),
    #     comm)
    # train_iterator = ch.iterators.SerialIterator(train, batch_size, shuffle=True)
    # vali_iterator = ch.iterators.SerialIterator(test, batch_size, repeat=False, shuffle=False)
    model = CosmoFlow(comm, comm, out)

    # model = L.Classifier(CosmoFlow(comm, out), lossfun=F.mean_squared_error)

    # print("Model Created successfully")
    ch.backends.cuda.get_device_from_id(device).use()
    model.to_gpu()  # Copy the model to the GPU

    # which optimiser should be used here, because we need gradient allreduce
    # optimizer = ch.optimizers.Adam()
    optimizer = chainermnx.create_spatial_optimizer(chainer.optimizers.Adam(), comm, out)
    optimizer.setup(model)
    # Create the updater, using the optimizer
    updater = chainermnx.training.StandardUpdater(train_iterator, optimizer, comm, out=out, device=device)
    # updater = training.StandardUpdater(train_iterator, optimizer, device=device)

    # Set up a trainer
    trainer = training.Trainer(updater, (epochs, 'iteration'), out=out)
    # trainer.extend(extensions.Evaluator(vali_iterator, model, device=device))

    log_interval = (1, 'iteration')
    filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".log"

    if comm.rank == 0:
        trainer.extend(extensions.DumpGraph('main/loss'))
        trainer.extend(extensions.LogReport(trigger=log_interval, filename=filename))
        # trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
        trainer.extend(extensions.PlotReport(
            ['main/loss', 'validation/main/loss'], 'epoch', filename='loss.png'))
        trainer.extend(extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'], 'epoch', filename='accuracy.png'))
        trainer.extend(extensions.ProgressBar())

    if comm.rank == 0:
        print("Starting training .....")

    trainer.run()
    if comm.rank == 0:
        print("Finished")


if __name__ == "__main__":

    main()


