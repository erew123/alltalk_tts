import datetime
import io
import logging

from matplotlib import pyplot as plt
from trainer import ConsoleLogger
from trainer.logging.console_logger import tcolors
from trainer.utils.distributed import rank_zero_only

logger = logging.getLogger("trainer")

class MetricsLogger(ConsoleLogger):
    def __init__(self, ):
        super().__init__()
        self.cur_epoch_start = None
        self.cur_epoch_end = None
        self.cur_epoch = None
        self.epoch_durations = []
        self.max_epoch = None
        self.batch_steps = None
        self.global_steps = None
        self.loss = []
        self.avg_loss = []
        self.loss_mel_ce = []
        self.avg_loss_mel_ce = []
        self.loss_text_ce = []
        self.avg_loss_text_ce = []
        self.learning_rates = []
        self.total_duration = None
        self.estimated_duration = None
        self.start = None

    @staticmethod
    def get_now():
        return datetime.datetime.now()
    @staticmethod
    def format_duration(seconds):
        if seconds:
            # Calculate hours, minutes, and seconds
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            seconds = seconds % 60
            return f"{int(hours)}:{int(minutes)}:{int(seconds)}"
        else:
            return None

    @rank_zero_only
    def print_epoch_start(self, epoch, max_epoch, output_path=None):
        self.cur_epoch = epoch
        self.cur_epoch_start = self.get_now()
        self.max_epoch = max_epoch
        super().print_epoch_start(epoch, max_epoch, output_path)

    @rank_zero_only
    def print_train_start(self):
        super().print_train_start()

    @rank_zero_only
    def print_train_step(self, batch_steps, step, global_step, loss_dict, avg_loss_dict):
        self.batch_steps = batch_steps
        self.global_steps = global_step

        for key, value in loss_dict.items():
            if key == 'loss_mel_ce':
                self.loss_mel_ce.append((global_step, value))
            elif key == 'loss':
                self.loss.append((global_step, value))
            elif key == 'loss_text_ce':
                self.loss_text_ce.append((global_step, value))
            elif key == 'current_lr':
                self.learning_rates.append((global_step, value))
        super().print_train_step(batch_steps, step, global_step, loss_dict, avg_loss_dict)

    # pylint: disable=unused-argument
    @rank_zero_only
    def print_train_epoch_end(self, global_step, epoch, epoch_time, print_dict):
        super().print_epoch_end(global_step, epoch, epoch_time, print_dict)
        if self.estimated_duration:
            self.log_with_flush(
                "\n{}{} >> ETA: {} {}".format(tcolors.UNDERLINE, tcolors.BOLD, self.format_duration(self.estimated_duration), tcolors.ENDC),
            )

    @rank_zero_only
    def print_eval_start(self):
        super().print_eval_start()

    @rank_zero_only
    def print_eval_step(self, step, loss_dict, avg_loss_dict):
        super().print_eval_step(step, loss_dict, avg_loss_dict)

    @rank_zero_only
    def print_epoch_end(self, epoch, avg_loss_dict):
        self.cur_epoch_end = self.get_now()
        self.epoch_durations.append((self.cur_epoch_end - self.cur_epoch_start).total_seconds())
        for key, value in avg_loss_dict.items():
            if key == 'avg_loss_mel_ce':
                self.avg_loss_mel_ce.append((epoch, value))
            elif key == 'avg_loss':
                self.avg_loss.append((epoch, value))
            elif key == 'avg_loss_text_ce':
                self.avg_loss_text_ce.append((epoch, value))

        if len(self.epoch_durations) > 1 and self.max_epoch:
            self.total_duration = sum(self.epoch_durations)
            additional_data_points_needed = self.max_epoch - len(self.epoch_durations)
            if additional_data_points_needed > 0:
                weighted_avg_duration = sum((i + 1) * duration for i, duration in enumerate(self.epoch_durations)) / sum(range(1, len(self.epoch_durations) + 1))
                self.estimated_duration = (weighted_avg_duration * additional_data_points_needed)

        super().print_epoch_end(epoch, avg_loss_dict)

    def plot_metrics(self, show_loss_mel_ce=False, show_loss_text_ce=False, show_loss=True):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(24, 18))

        ax2.set_title('Steps')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Value')

        ax1.set_title('Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Value')

        ax3.set_title('Learning Rates')
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Value')

        if self.loss and self.loss_text_ce and self.loss_mel_ce:
            if show_loss:
                ax2.plot(*zip(*self.loss), color='green', linestyle='-', label='Loss')
                ax2.fill_between(*zip(*self.loss), color='lightgreen', alpha=0.1)
            if show_loss_text_ce:
                ax2.plot(*zip(*self.loss_text_ce), color='red', linestyle='-', label='Loss Text')
                ax2.fill_between(*zip(*self.loss_text_ce), color='lightcoral', alpha=0.1)
            if show_loss_mel_ce:
                ax2.plot(*zip(*self.loss_mel_ce), color='blue', linestyle='-', label='Loss MEL')
                ax2.fill_between(*zip(*self.loss_mel_ce), color='lightblue', alpha=0.1)
            ax2.legend()

        if self.learning_rates:
            ax3.plot(*zip(*self.learning_rates), linestyle='-', color='blue', label='Learning Rate')
            ax3.fill_between(*zip(*self.learning_rates), color='lightblue', alpha=0.1)
            ax3.legend()

        if self.avg_loss and self.avg_loss_text_ce and self.avg_loss_mel_ce:
            if show_loss:
                ax1.plot(*zip(*self.avg_loss), color='green', linestyle='-', label='Avg Loss')
                ax1.fill_between(*zip(*self.avg_loss), color='lightgreen', alpha=0.1)
            if show_loss_text_ce:
                ax1.plot(*zip(*self.avg_loss_text_ce), color='red', linestyle='-', label='Avg Loss Text')
                ax1.fill_between(*zip(*self.avg_loss_text_ce), color='lightcoral', alpha=0.1)
            if show_loss_mel_ce:
                ax1.plot(*zip(*self.avg_loss_mel_ce), color='blue', linestyle='-', label='Avg Loss MEL')
                ax1.fill_between(*zip(*self.avg_loss_mel_ce), color='lightblue', alpha=0.1)
            ax1.set_xlim(left=0)
            ax1.legend()

        plt.style.use('dark_background')
        plt.tight_layout()
        plt.savefig("training_metrics.png", format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close()
        return "training_metrics.png"