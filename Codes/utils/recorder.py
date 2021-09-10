"""
A class to incorprate all necessary function to record training log
"""
import time
import os
import shutil

import numpy as np

from utils.train import is_int, progress_bar, accuracy, AverageMeter

class Recorder():
    """
    A class to record training log and write into txt file
    """

    def __init__(self, SummaryPath, dataset_name='CIFAR10', task_name = None, start_epoch = 0, epoch_size = 391):

        self.SummaryPath = SummaryPath

        if not os.path.exists(SummaryPath):
            os.makedirs(SummaryPath)
        else:
            if start_epoch != 0:
                print('Resume training from epoch %d' %start_epoch)
            else:
                print('Record exist, remove')
                shutil.rmtree(SummaryPath)
                os.makedirs(SummaryPath)

        print('Summary records saved at: %s' %SummaryPath)

        self.task_name = task_name
        if self.task_name is None:
            prefix = ''
        else:
            prefix = '%s-' %self.task_name

        self.dataset_type = 'large' if dataset_name in ['ImageNet'] else 'small'

        ##########
        # Shared #
        ##########
        # For shared
        self.train_loss = 0
        self.niter = start_epoch * epoch_size  # Overall iteration record
        self.test_loss = 0
        self.smallest_training_loss = 1e9
        self.stop = False  # Whether to stop training
        self.epoch = 0
        self.best_test_flag = False
        self.constraint = AverageMeter()

        # For CIFAR dataset
        # self.train_acc = AverageMeter()
        self.loss = AverageMeter()
        self.total = 0  # Number of instance used in training
        self.n_batch = 0  # Number of batches used in training
        self.test_acc = 0
        self.best_test_acc = 0
        self.ascend_count = 0

        # For ImageNet dataset
        self.loss_ImageNet = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.test_acc_top1 = 0
        self.test_acc_top5 = 0
        self.best_test_acc_top1 = 0
        self.best_test_acc_top5 = 0

        ###################
        # Initialize file #
        ###################
        if self.dataset_type == 'small':
            self.loss_record = open('%s/%sloss.txt' % (self.SummaryPath, prefix), 'a+')
            self.train_acc_record = open('%s/%strain-acc.txt' % (self.SummaryPath, prefix), 'a+')
            self.test_acc_record = open('%s/%stest-acc.txt' % (self.SummaryPath, prefix), 'a+')
            self.lr_record = open('%s/%slr.txt' % (self.SummaryPath, prefix), 'a+')
            self.constraint_record = open('%s/%sconstraint.txt' % (self.SummaryPath, prefix), 'a+')
        else:
            self.loss_record = open('%s/%sloss.txt' % (self.SummaryPath, prefix), 'a+')
            self.train_top1_acc_record = open('%s/%strain-top1-acc.txt' % (self.SummaryPath, prefix), 'a+')
            self.train_top5_acc_record = open('%s/%strain-top5-acc.txt' % (self.SummaryPath, prefix), 'a+')
            self.test_top1_acc_record = open('%s/%stest-top1-acc.txt' % (self.SummaryPath, prefix), 'a+')
            self.test_top5_acc_record = open('%s/%stest-top5-acc.txt' % (self.SummaryPath, prefix), 'a+')
            self.lr_record = open('%s/%slr.txt' % (self.SummaryPath, prefix), 'a+')

        self.args_record = open('%s/arguments.txt' % (self.SummaryPath), 'w+')

    def write_arguments(self, args_list):

        for args in args_list:
            if isinstance(args, dict):
                for key, value in args.items():
                    self.args_record.write('%s: %s\n' %(key, value))
            else:
                for arg in vars(args):
                    self.args_record.write('%s: %s\n' %(arg, getattr(args, arg)))

            self.flush(self.args_record)

        self.args_record.close()

    def update(self, loss=0, acc: list=[0, 0], constraint=0, batch_size=0, cur_lr=1e-3, end=None, is_train=True):

        if is_train:

            self.train_loss += loss
            self.n_batch += 1
            self.total += batch_size
            self.niter += 1

            if self.dataset_type == 'small':
                self.loss.update(loss, batch_size)
                self.top1.update(acc[0], batch_size)
                self.constraint.update(constraint, batch_size)
                # self.batch_time.update(time.time() - end)
                # self.loss_record.write('%d, %.8f\n' % (self.niter, self.train_loss / self.n_batch))
                self.loss_record.write('%d, %.8f, %.8f\n'% (self.niter, self.loss.val, self.loss.avg))
                self.train_acc_record.write('%d, %.4f, %.4f\n'% (self.niter, self.top1.val, self.top1.avg))
                self.lr_record.write('%d, %e\n' % (self.niter, cur_lr))
                self.constraint_record.write('%d, %.8f, %.8f\n' % (self.niter, self.constraint.val, self.constraint.avg))

                self.flush([self.loss_record, self.train_acc_record, self.lr_record, self.constraint_record])

            else:
                self.batch_time.update(time.time() - end)
                self.top1.update(acc[0], batch_size)
                self.top5.update(acc[1], batch_size)
                self.loss_ImageNet.update(loss, batch_size)

                self.loss_record.write('%d, %.8f\n' % (self.niter, self.loss_ImageNet.avg))
                self.train_top1_acc_record.write('%d, %.4f\n' % (self.niter, self.top1.avg))
                self.train_top5_acc_record.write('%d, %.4f\n' % (self.niter, self.top5.avg))
                self.lr_record.write('%d, %e\n' % (self.niter, cur_lr))

                self.flush([self.loss_record, self.train_top1_acc_record, self.train_top5_acc_record, self.lr_record])

        else:
            if self.dataset_type == 'small':

                if isinstance(acc, float):
                    self.test_acc = acc

                    if self.best_test_acc < self.test_acc:
                        self.best_test_acc = self.test_acc
                        print('Best test acc')
                        self.best_test_flag = True
                        # self.save(self.SummaryPath)
                    else:
                        self.best_test_flag = False

                    self.test_acc_record.write('%d, %.4f\n' % (self.niter, self.test_acc))

                elif isinstance(acc, list):
                    self.test_acc = np.mean(acc)
                    self.test_acc_record.write('%d' %self.niter)
                    for each_acc in acc:
                        self.test_acc_record.write(', %.4f' % each_acc)
                    self.test_acc_record.write('\n')

                self.flush([self.test_acc_record])

            else:
                # pass
                self.test_acc_top1, self.test_acc_top5 = acc[0], acc[1]

                if self.best_test_acc_top1 < self.test_acc_top1 or self.best_test_acc_top5 < self.test_acc_top5:
                    self.best_test_acc_top1 = self.test_acc_top1
                    self.best_test_acc_top5 = self.test_acc_top5
                    self.best_test_flag = True
                    print('Best test acc')
                    # self.save(self.SummaryPath)
                else:
                    self.best_test_flag = False

                self.test_top1_acc_record.write('%d, %.3f\n' % (self.niter, self.test_acc_top1))
                self.test_top5_acc_record.write('%d, %.3f\n' % (self.niter, self.test_acc_top5))

                self.flush([self.test_top1_acc_record, self.test_top5_acc_record])


    def restart_training(self):
        self.reset_performance()
        self.reset_best_test_acc()
        self.stop = False


    def reset_performance(self):

        self.train_loss = 0
        self.epoch += 1

        if self.dataset_type == 'small':
            self.loss.reset()
            self.top1.reset()
            self.constraint.reset()
            self.total = 0
            self.n_batch = 0
        else:
            self.loss_ImageNet.reset()
            self.top1.reset()
            self.top5.reset()
            self.batch_time.reset()


    def flush(self, file_list=None):

        for file in file_list:
            file.flush()


    def close(self):

        if self.dataset_type == 'small':
            self.loss_record.close()
            self.train_acc_record.close()
            self.test_acc_record.close()
            # self.lr_record.close()
            self.constraint_record.close()
        else:
            self.loss_record.close()
            self.train_top1_acc_record.close()
            self.train_top5_acc_record.close()
            self.test_top1_acc_record.close()
            self.test_top5_acc_record.close()
            self.lr_record.close()


    def get_best_test_acc(self):

        if self.dataset_type == 'small':
            print('Best test acc: %.3f' %self.best_test_acc)
            return self.best_test_acc
        else:
            print('Best test top1 acc: %.3f, top5 acc: %.3f'
                  % (self.best_test_acc_top1, self.best_test_acc_top5))
            return (self.best_test_acc_top1, self.best_test_acc_top5)

    def reset_best_test_acc(self):

        if self.dataset_type == 'small':
            self.best_test_acc = 0
        else:
            self.best_test_acc_top1 = 0
            self.best_test_acc_top5 = 0


    def update_smallest_train_loss(self):
        self.smallest_training_loss = self.train_loss
        print('Current smallest training loss: %.3f' %self.smallest_training_loss)


    def adjust_lr(self, optimizer, adjust_type='dorefa', epoch=-1, init_lr=1e-3):
        """
        Adjust learning rate
        :param optimizer:
        :param adjust_type:
        :param epoch:
        :return:
        """

        change_flag = False

        if self.train_loss < self.smallest_training_loss:
            self.smallest_training_loss = self.train_loss
            print('Current smallest training loss: %.3f' % self.smallest_training_loss)
            self.ascend_count = 0
        else:
            self.ascend_count += 1
            print('Training loss: %.3f [%.3f], ascend count: %d'
                  %(self.train_loss, self.smallest_training_loss, self.ascend_count))

        if self.dataset_type == 'small':
            if adjust_type == 'adaptive':
                if self.ascend_count >= 3:
                    self.ascend_count = 0
                    change_flag = True
                    optimizer.param_groups[0]['lr'] *= 0.1
                    print('>>>>>>>>>> [%s] Learning rate change to %e <<<<<<<<<<<<<' % \
                          (self.task_name, optimizer.param_groups[0]['lr']))
                    if optimizer.param_groups[0]['lr'] < (init_lr * 1e-3):
                        self.stop = True

            elif is_int(adjust_type):
                n_change_epoch = int(adjust_type)
                if (epoch+1) % n_change_epoch == 0:
                    change_flag = True
                    optimizer.param_groups[0]['lr'] *= 0.1
                    print('>>>>>>>>>> [%s] Learning rate change to %e <<<<<<<<<<<<<' % \
                          (self.task_name, optimizer.param_groups[0]['lr']))
                    if optimizer.param_groups[0]['lr'] < (init_lr * 1e-3):
                        self.stop = True

            else:
                raise NotImplementedError

        elif self.dataset_type == 'large':

            if (epoch + 1) % 10 == 0:
                change_flag = True
                optimizer.param_groups[0]['lr'] *= 0.1
                print('>>>>>>>>>> [%s] Learning rate change to %e <<<<<<<<<<<<<' % \
                      (self.task_name, optimizer.param_groups[0]['lr']))

        return change_flag


    def print_training_result(self, batch_idx, n_batch, monitor_freq=100, append=""):

        if self.dataset_type == 'small':
            progress_bar(batch_idx, n_batch, "Loss: %.3f (%.3f), Acc: %.3f%% (%.3f%%) | %s"
                         % (self.loss.val, self.loss.avg,  self.top1.val, self.top1.avg, append))
        else:
            # raise NotImplementedError
            if batch_idx % monitor_freq == 0:
                print('Training: [%d / %d] \t Time %.3f (%.3f) \t  Loss %.4f(%.4f)\n'
                      'Prec@1 %.4f(%.4f) \t Prec@5 %.4f(%.4f) \n'\
                      %(batch_idx, n_batch, self.batch_time.val, self.batch_time.sum,
                        self.loss_ImageNet.val, self.loss_ImageNet.avg,
                        self.top1.val, self.top1.avg, self.top5.val, self.top5.avg))
                if append is not None:
                    print(append+'\n')


if __name__ == '__main__':


    recorder = Recorder('./Results/test', dataset_name='ImageNet')
    recorder.reset_performance()
    recorder.update(loss=1.0, acc=(99, 89), end=time.time())
    recorder.print_training_result(0, 100)





