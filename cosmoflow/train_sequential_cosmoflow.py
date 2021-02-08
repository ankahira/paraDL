import os
import chainer as ch
import chainermn
from chainer import datasets, training
from chainer.training import extensions
import argparse
import chainermnx
import chainer
from datetime import datetime

import shutil
import os


# Local imports
from models.sequential_cosmoflow import CosmoFlow
from utils.cosmoflow_data_prep import CosmoDataset
from chainer.function_hooks import TimerHook


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
    parser = argparse.ArgumentParser(description='CosmoFlow Multi-Node Training')
    parser.add_argument('--batchsize', '-B', type=int, default=32, help='Learning minibatch size')
    parser.add_argument('--epochs', '-E', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--out', '-o', default='results', help='Output directory')
    parser.add_argument('--gpu', '-g', type=int,  default=0, help='GPU ID (negative value indicates CPU)')

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

    device = args.gpu
    train = CosmoDataset("/groups2/gaa50004/cosmoflow_data")
    # Use the modified multinode iterator that makes sure Y has the correct values
    train_iterator = chainer.iterators.MultithreadIterator(train, batch_size, n_threads=80, shuffle=True)

    model = CosmoFlow()

    ch.backends.cuda.get_device_from_id(device).use()
    model.to_gpu()  # Copy the model to the GPU

    optimizer =chainer.optimizers.Adam()

    optimizer.setup(model)
    # Create the updater, using the optimizer
    updater = training.StandardUpdater(train_iterator, optimizer, device=device)

    # Set up a trainer
    trainer = training.Trainer(updater, (epochs, 'iteration'), out=out)
    # trainer.extend(extensions.Evaluator(vali_iterator, model, device=device))

    log_interval = (1, 'iteration')
    filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".log"

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

    print("Starting training .....")

    hook = TimerHook()
    time_hook_results_file = open(os.path.join(args.out, "function_times.txt"), "a")

    with hook:
        trainer.run()

        print("Finished")

    hook.print_report()
    hook.print_report(file=time_hook_results_file)


if __name__ == "__main__":
    main()
