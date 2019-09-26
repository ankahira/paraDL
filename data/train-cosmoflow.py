import chainer as ch
import chainermn
from chainer import datasets, training
from chainer.training import extensions
import argparse
# Local imports
from models import CosmoFlowMP
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
    training_data = CosmoDataset("/home/acb10954wf/data")  ## actual data
    # training_data = temp_data_prep()  # temp data
    print("Fetching data successful")
    print("Found %d training samples" % training_data.__len__())
    train, test = datasets.split_dataset_random(
        training_data, first_size=(int(training_data.__len__() * 0.80)))
    train_iterator = ch.iterators.SerialIterator(train, 1)
    vali_iterator = ch.iterators.SerialIterator(test, 1, repeat=False, shuffle=False)

    model = CosmoFlowMP(comm)

    print("Model Created successfully")

    if args.gpu:
        # Make a specified GPU current
        ch.backends.cuda.get_device_from_id(device).use()
        model.to_gpu()  # Copy the model to the GPU

    optimizer = ch.optimizers.Adam()

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


