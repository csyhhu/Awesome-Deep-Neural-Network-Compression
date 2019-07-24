"""
A class to incorprate all necessary function to record training log
"""
import time
from utils.train import AverageMeter, accuracy, progress_bar

class Recorder():
    """
    A class to record training log and write into txt file
    """

    def __init__(self, SummaryPath, dataset_name, task_name = None):

        self.SummaryPath = SummaryPath

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
        self.niter = 0  # Overall iteration record
        self.test_loss = 0
        self.smallest_training_loss = 1e9
        self.stop = False  # Whether to stop training
        self.epoch = 0

        # For CIFAR dataset
        # self.train_acc = AverageMeter()
        self.total = 0  # Number of batches used in training
        self.n_batch = 0  # Number of batches used in training
        self.test_acc = 0
        self.best_test_acc = 0
        self.ascend_count = 0

        # For ImageNet dataset
        # self.loss = AverageMeter()
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
            self.loss_record = open('%s/%sloss.txt' % (self.SummaryPath, prefix), 'w+')
            self.train_acc_record = open('%s/%strain-acc.txt' % (self.SummaryPath, prefix), 'w+')
            self.test_acc_record = open('%s/%stest-acc.txt' % (self.SummaryPath, prefix), 'w+')
            self.lr_record = open('%s/%slr.txt' % (self.SummaryPath, prefix), 'w+')
        else:
            self.loss_record = open('%s/%sloss.txt' % (self.SummaryPath, prefix), 'w+')
            self.train_top1_acc_record = open('%s/%strain-top1-acc.txt' % (self.SummaryPath, prefix), 'w+')
            self.train_top5_acc_record = open('%s/%strain-top5-acc.txt' % (self.SummaryPath, prefix), 'w+')
            self.test_top1_acc_record = open('%s/%stest-top1-acc.txt' % (self.SummaryPath, prefix), 'w+')
            self.test_top5_acc_record = open('%s/%stest-top5-acc.txt' % (self.SummaryPath, prefix), 'w+')
            self.lr_record = open('%s/%slr.txt' % (self.SummaryPath, prefix), 'w+')

    def update(self, loss, acc, batch_size=0, cur_lr=1e-3, end=None, is_train = True):

        if is_train:

            self.train_loss += loss
            self.n_batch += 1
            self.total += batch_size
            self.niter += 1

            if self.dataset_type == 'small':
                self.top1.update(acc[0], batch_size)

                self.loss_record.write('%d, %.8f\n' % (self.niter, self.train_loss / self.n_batch))
                self.train_acc_record.write('%d, %.3f\n' % (self.niter, self.top1.avg))
                self.lr_record.write('%d, %e\n' % (self.niter, cur_lr))

                self.flush([self.loss_record, self.train_acc_record, self.lr_record])

            else:
                self.batch_time.update(time.time() - end)
                self.top1.update(acc[0], batch_size)
                self.top5.update(acc[1], batch_size)

                self.loss_record.write('%d, %.8f\n' % (self.niter, self.train_loss / self.n_batch))
                self.train_top1_acc_record.write('%d, %.3f\n' % (self.niter, self.top1.avg))
                self.train_top5_acc_record.write('%d, %.3f\n' % (self.niter, self.top5.avg))
                self.lr_record.write('%d, %ef\n' % (self.niter, cur_lr))

                self.flush([self.loss_record, self.train_top1_acc_record, self.train_top5_acc_record, self.lr_record])

        else:

            if self.dataset_type == 'small':

                self.test_acc = acc

                if self.best_test_acc < self.test_acc:
                    self.best_test_acc = self.test_acc
                    print('Best test acc')
                    # self.save(self.SummaryPath)

                self.test_acc_record.write('%d, %.3f\n' % (self.niter, self.test_acc))
                self.flush([self.test_acc_record])

            else:
                pass
                # self.test_acc_top1, self.test_acc_top5 = acc[0], acc[1]
                #
                # if self.best_test_acc_top1 < self.test_acc_top1 or self.best_test_acc_top5 < self.test_acc_top5:
                #     self.best_test_acc_top1 = self.test_acc_top1
                #     self.best_test_acc_top5 = self.test_acc_top5
                #     print('Best test acc')
                #     # self.save(self.SummaryPath)
                #
                # self.test_top1_acc_record.write('%d, %.3f\n' % (self.niter, self.test_acc_top1))
                # self.test_top5_acc_record.write('%d, %.3f\n' % (self.niter, self.test_acc_top5))
                #
                # self.flush([self.test_top1_acc_record, self.test_top5_acc_record])


    def reset_performance(self):

        self.train_loss = 0
        self.epoch += 1

        if self.dataset_type == 'small':
            self.top1.reset()
            self.total = 0
            self.n_batch = 0
        else:
            # self.best_test_acc_top1 = 0
            # self.best_test_acc_top5 = 0
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
            self.lr_record.close()
        else:
            self.loss_record.close()
            self.train_top1_acc_record.close()
            self.train_top5_acc_record.close()
            self.test_top1_acc_record.close()
            self.test_top5_acc_record.close()
            self.lr_record.close()


    def get_best_test_acc(self):

        if self.dataset_type == 'small':
            return self.best_test_acc
        else:
            return self.best_test_acc_top1, self.best_test_acc_top5

    def reset_best_test_acc(self):

        if self.dataset_type == 'small':
            self.best_test_acc = 0
        else:
            self.best_test_acc_top1 = 0
            self.best_test_acc_top5 = 0


    def restart_training(self):
        """
        Restart another training by resetting best test acc, optimizer
        :return:
        """
        self.reset_best_test_acc()
        self.reset_performance()


    def update_smallest_train_loss(self):
        self.smallest_training_loss = self.train_loss
        print('Current smallest training loss: %.3f' %self.smallest_training_loss)


    def adjust_lr(self, optimizer):

        if self.train_loss < self.smallest_training_loss:
            self.smallest_training_loss = self.train_loss
            print('Current smallest training loss: %.3f' % self.smallest_training_loss)
            self.ascend_count = 0
        else:
            self.ascend_count += 1
            print('Training loss: %.3f [%.3f], ascend count: %d'
                  %(self.train_loss, self.smallest_training_loss, self.ascend_count))

        if self.dataset_type == 'small' and self.ascend_count >= 3:
            self.ascend_count = 0
            optimizer.param_groups[0]['lr'] *= 0.1
            print('>>>>>>>>>> [%s] Learning rate change to %e <<<<<<<<<<<<<' % \
                  (self.task_name, optimizer.param_groups[0]['lr']))
            if optimizer.param_groups[0]['lr'] < 1e-6:
                self.stop = True

        elif self.dataset_type == 'large':
            """
            To DO: How to adjust learning rate in ImageNet dataset model
            """
            pass

            # if self.epoch % 10 == 0:
            #     optimizer.param_groups[0]['lr'] *= 0.1

            # raise NotImplementedError


    def print_training_result(self, batch_idx, n_batch, append=""):

        if self.dataset_type == 'small':
            progress_bar(batch_idx, n_batch,
                         "Loss: %.3f, Acc: %.3f%% | %s"
                         % (self.train_loss / (batch_idx + 1), self.top1.avg, append))
        else:
            raise NotImplementedError




