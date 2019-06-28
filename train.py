import chainer as ch
from chainer import datasets, training
from chainer.training import extensions

# Local imports
from models.CosmoFlow import CosmoFlow
from extras import temp_data_prep
from data.data import CosmoDataset

import matplotlib

matplotlib.use('Agg')


def main():
    # Input data and label

    # training_data = CosmoDataset("data/training") // actual data
    training_data = temp_data_prep() ## temp data
    print("Fetching data successful")
    train, test = datasets.split_dataset_random(training_data, first_size=(int(training_data.__len__() * 0.80)))

    # train_iterator = ch.iterators.SerialIterator(training_data, 1)
    train_iterator = ch.iterators.SerialIterator(train, 1)
    vali_iterator = ch.iterators.SerialIterator(test, 1, repeat=False, shuffle=False)

    model = CosmoFlow()

    print("Model Created successfully")

    gpu_id = -1  # Set to -1 if you use CPU
    if gpu_id >= 0:
        model.to_gpu(gpu_id)

    optimizer = ch.optimizers.Adam()

    optimizer.setup(model)
    # Create the updater, using the optimizer
    updater = training.StandardUpdater(train_iterator, optimizer, device=gpu_id)

    # Set up a trainer
    trainer = training.Trainer(updater, (10, 'epoch'), out='result')

    log_interval = (1, 'epoch')

    trainer.extend(extensions.Evaluator(vali_iterator, model, device=gpu_id))

    trainer.extend(extensions.DumpGraph('main/loss'))
    # trainer.extend(extensions.snapshot())
    # trainer.extend(extensions.snapshot_object(model, 'model_iter_{.updater.iteration}'))
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(['epoch' 'Validation loss', 'lr']), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    print("Starting Training ")
    trainer.run()


if __name__ == "__main__":

    main()


