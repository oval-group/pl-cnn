import time

from config import Configuration as Cfg


def print_obj_and_acc(objective, accuracy, which_set):

    objective_str = '{} objective:'.format(which_set.title())
    accuracy_str = '{} accuracy:'.format(which_set.title())
    print("{:32} {:.5f}".format(objective_str, objective))
    print("{:32} {:.2f}%".format(accuracy_str, accuracy))


def performance(nnet, which_set, print_=False):

    objective = 0
    accuracy = 0
    batches = 0
    for batch in nnet.data.get_epoch(which_set):
        inputs, targets, _ = batch
        err, acc = nnet.forward(inputs, targets)
        objective += err
        accuracy += acc
        batches += 1

    objective /= batches
    accuracy *= 100. / batches

    if print_:
        print_obj_and_acc(objective, accuracy, which_set)

    return objective, accuracy


def performance_val_avg(nnet, layer):

    C = Cfg.C.get_value()

    val_objective = 0
    val_accuracy = 0

    for batch in nnet.data.get_epoch_val():
        inputs, targets, _ = batch
        err, acc = layer.hinge_avg(inputs, targets)
        val_objective += err
        val_accuracy += acc

    val_objective /= nnet.data.n_val
    val_accuracy *= 100. / nnet.data.n_val

    val_objective += layer.norm_avg() / (2. * C)
    for other_layer in nnet.trainable_layers:
        if other_layer is not layer:
            val_objective += other_layer.norm() / (2. * C)

    return val_objective, val_accuracy


def hinge_avg(nnet, layer):

    train_objective = 0
    train_accuracy = 0
    for batch in nnet.data.get_epoch_train():
        inputs, targets, _ = batch
        new_err, new_acc = layer.hinge_avg(inputs, targets)

        train_objective += new_err
        train_accuracy += new_acc

    train_objective *= 1. / nnet.data.n_train
    train_accuracy *= 100. / nnet.data.n_train

    return train_objective, train_accuracy


def primal_avg(nnet, layer):

    train_objective, train_accuracy = hinge_avg(nnet, layer)

    train_objective += layer.warm_reg_avg()

    return train_objective, train_accuracy


def performance_train_avg(nnet, layer):

    C = Cfg.C.get_value()

    train_objective, train_accuracy = hinge_avg(nnet, layer)

    train_objective += layer.norm_avg() / (2. * C)

    for llayer in nnet.trainable_layers:
        if llayer is not layer:
            train_objective += llayer.norm() / (2. * C)

    return train_objective, train_accuracy


def checkpoint(nnet, layer):

    if nnet.data.dataset_name == "imagenet":
        return

    C = Cfg.C.get_value()

    hinge_loss, train_accuracy = hinge_avg(nnet, layer)

    primal_objective = hinge_loss + layer.warm_reg_avg()

    train_objective = hinge_loss + layer.norm_avg() / (2. * C)

    for other_layer in nnet.trainable_layers:
        if other_layer is not layer:
            train_objective += other_layer.norm() / (2. * C)

    dual_objective = layer.get_dual()

    val_objective, val_accuracy = performance_val_avg(nnet, layer)

    train_objective = float(train_objective)
    train_accuracy = float(train_accuracy)
    val_objective = float(val_objective)
    val_accuracy = float(val_accuracy)
    primal_objective = float(primal_objective)
    hinge_loss = float(hinge_loss)
    dual_objective = float(dual_objective)

    t = time.time() - nnet.clock

    nnet.log['time_stamp'].append(t)
    nnet.log['layer_tag'].append(layer.name)

    nnet.log['train_objective'].append(train_objective)
    nnet.log['train_accuracy'].append(train_accuracy)

    nnet.log['val_objective'].append(val_objective)
    nnet.log['val_accuracy'].append(val_accuracy)

    nnet.log['primal_objective'].append(primal_objective)
    nnet.log['hinge_loss'].append(hinge_loss)
    nnet.log['dual_objective'].append(dual_objective)

    print("Pass %i - Epoch %i - Layer %s" %
          (nnet.pass_, len(nnet.log['time_stamp']), layer.name))
    print_obj_and_acc(train_objective, train_accuracy, which_set='train')
    print_obj_and_acc(val_objective, val_accuracy, which_set='val')

    if Cfg.draw_on_board:
        data_dict = {'Objective': train_objective,
                     'Accuracy': train_accuracy,
                     'Primal Objective': primal_objective,
                     'Hinge Loss': hinge_loss,
                     'Dual Objective': dual_objective}
        nnet.board_monitor_train.add_scalar_dict(data_dict)
        data_dict = {'Objective': val_objective,
                     'Accuracy': val_accuracy}
        nnet.board_monitor_val.add_scalar_dict(data_dict)


def check_dual(nnet, layer):

    dual_objective = layer.get_dual()
    t = time.time() - nnet.clock
    nnet.log['dual_objective'].append(dual_objective)

    nnet.log['time_stamp'].append(t)
    nnet.log['layer_tag'].append(layer.name)
