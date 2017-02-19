import time

from utils.monitoring import performance, print_obj_and_acc
from config import Configuration as Cfg


def train_network(nnet):

    if nnet.data.dataset_name == "imagenet":
        train_network_imagenet(nnet)
        return

    print("Starting training with %s" % nnet.sgd_solver)

    for epoch in range(nnet.n_epochs):

        if nnet.solver == 'nesterov' and (epoch + 1) % 10 == 0:
            lr = Cfg.floatX(Cfg.learning_rate.get_value() / 10.)
            Cfg.learning_rate.set_value(lr)

        # In each epoch, we do a full pass over the training data:
        train_objective = 0
        train_accuracy = 0
        train_batches = 0
        start_time = time.time()

        # train on epoch
        for batch in nnet.data.get_epoch_train():
            inputs, targets, _ = batch
            err, acc = nnet.backprop(inputs, targets)
            train_objective += err
            train_accuracy += acc
            train_batches += 1

        # normalize results
        train_objective *= 1. / train_batches
        train_accuracy *= 100. / train_batches

        # print performance
        print_obj_and_acc(train_objective, train_accuracy, which_set='train')
        val_objective, val_accuracy = performance(nnet, which_set='val',
                                                  print_=True)
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, nnet.n_epochs, time.time() - start_time))

        # log performance
        nnet.log['train_objective'].append(train_objective)
        nnet.log['train_accuracy'].append(train_accuracy)
        nnet.log['val_objective'].append(val_objective)
        nnet.log['val_accuracy'].append(val_accuracy)
        nnet.log['time_stamp'].append(time.time() - nnet.clock)

        # send data to Tensorboard
        if Cfg.draw_on_board:
            data_to_send = {'Objective': train_objective,
                            'Accuracy': train_accuracy}
            nnet.board_monitor_train.add_scalar_dict(data_to_send)
            data_to_send = {'Objective': val_objective,
                            'Accuracy': val_accuracy}
            nnet.board_monitor_val.add_scalar_dict(data_to_send)

        # save model as required
        if epoch + 1 == nnet.save_at:
            nnet.dump_weights(nnet.save_to)

    test_objective, test_accuracy = performance(nnet, which_set='test',
                                                print_=True)

    # log final performance
    nnet.log['test_objective'] = test_objective
    nnet.log['test_accuracy'] = test_accuracy
    nnet.test_time = time.time() - nnet.clock


def train_network_imagenet(nnet):

    assert nnet.data.dataset_name == "imagenet"

    # Launch the training loop.
    print("Starting training with %s" % nnet.sgd_solver)
    # We iterate over epochs:
    for epoch in range(nnet.n_epochs):

        # In each epoch, we do a full pass over the training data:
        train_objective = 0
        train_accuracy = 0
        train_batches = 0

        for batch in nnet.data.get_epoch_train():
            inputs, targets, _ = batch
            err, acc = nnet.backprop(inputs, targets)
            train_objective += err
            train_accuracy += acc
            train_batches += 1

            if train_batches % 50 == 0:
                print("  training loss:\t\t{:.6f}"
                      .format(train_objective / train_batches))
                print("  training accuracy:\t\t{:.2f} %"
                      .format(train_accuracy * 100. / train_batches))

        nnet.log['train_objective'].append(train_objective / train_batches)
        nnet.log['train_accuracy'].append(train_accuracy * 100. /
                                          train_batches)

        nnet.log['time_stamp'].append(time.time() - nnet.clock)

        nnet.dump_weights("../log/saved_models/imagenet_%s_%i.p"
                          % (nnet.solver, epoch))
