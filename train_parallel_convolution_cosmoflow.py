import chainer as ch
import chainermn
from chainer import datasets, training
from chainer.training import extensions
import argparse

# Local imports
from models import ParallelConvolutionCosmoFlow
from extras import temp_data_prep
from data.data import CosmoDataset

import matplotlib

matplotlib.use('Agg')


def main():
    parser = argparse.ArgumentParser(description='CosmoFlow Multi-Node Training')
    parser.add_argument('--gpu', '-g', action='store_true', default=True, help='use GPU')
    args = parser.parse_args()

    #  Create ChainerMN communicator.
    if args.gpu:
        comm = chainermn.create_communicator('hierarchical')
        device = comm.rank
    else:
        comm = chainermn.create_communicator('naive')
        device = -1

    # Input data and label
    # training_data = CosmoDataset("/home/albert/data/training")  ## actual data
    training_data = temp_data_prep() ## temp data
    print("Fetching data successful")
    print("Found %d training samples" % training_data.__len__())
    train, test = datasets.split_dataset_random(training_data, first_size=(int(training_data.__len__() * 0.80)))

    # Set up Model
    model = ParallelConvolutionCosmoFlow(comm)

    if args.gpu:
        # Make a specified GPU current
        ch.backends.cuda.get_device_from_id(device).use()
        model.to_gpu()  # Copy the model to the GPU

    # Set the optimiser
    optimizer = ch.optimizers.Adam()
    optimizer.setup(model)

    if comm.rank != 0:
        train = chainermn.datasets.create_empty_dataset(train)
        test = chainermn.datasets.create_empty_dataset(test)

    train_iterator = chainermn.iterators.create_multi_node_iterator(ch.iterators.SerialIterator(train, 1), comm)
    vali_iterator = chainermn.iterators.create_multi_node_iterator(ch.iterators.SerialIterator(test, 1, repeat=False, shuffle=False), comm)

    # Create the updater, using the optimizer
    updater = training.StandardUpdater(train_iterator, optimizer, device=device)

    # Set up a trainer
    trainer = training.Trainer(updater, (10, 'epoch'), out='result')
    trainer.extend(extensions.Evaluator(vali_iterator, model, device=device))

    log_interval = (1, 'epoch')
    if comm.rank == 0:
        trainer.extend(extensions.DumpGraph('main/loss'))
        # trainer.extend(extensions.snapshot())
        # trainer.extend(extensions.snapshot_object(model, 'model_iter_{.updater.iteration}'))
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.PrintReport(['epoch' 'Validation loss', 'lr']), trigger=log_interval)
        trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.run()


if __name__ == "__main__":

    main()


