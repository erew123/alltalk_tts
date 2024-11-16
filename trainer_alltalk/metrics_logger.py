import datetime
import os
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
        self.current_model_path = None

    @staticmethod
    def get_now():
        return datetime.datetime.now()
    @staticmethod
    def format_duration(seconds):
        if seconds:
            # Calculate hours, minutes, and seconds
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            return f"{hours:02d}h {minutes:02d}m {seconds:02d}s"  # More readable format
        else:
            return "N/A"  # Better than None for display

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

    def update_model_path(self, path):
        """Set current model path for log reading"""
        try:
            # Find all XTTS_FT folders in the given path
            training_folders = []
            for folder in os.listdir(path):
                full_path = os.path.join(path, folder)
                if os.path.isdir(full_path) and folder.startswith("XTTS_FT"):
                    training_folders.append(full_path)

            if not training_folders:
                print(f"[DEBUG] No XTTS_FT folders found in {path}")
                self.current_model_path = path
                return

            # Get the most recent folder
            latest_folder = max(training_folders, key=os.path.getctime)
            # print(f"[DEBUG] Found latest training folder: {latest_folder}")
            self.current_model_path = latest_folder
            self.global_steps = 0  # Reset global steps

        except Exception as e:
            print(f"[DEBUG] Error updating model path: {e}")
            self.current_model_path = path

    def _format_metrics_for_log(self):
        """Format metrics for our log file"""
        log_text = "\n=== Training Metrics Update ===\n"
        log_text += f"Time: {self.get_now()}\n"
        
        if self.avg_loss:
            log_text += f"Current Average Loss: {self.avg_loss[-1][1]}\n"
        if self.avg_loss_mel_ce:
            log_text += f"Current MEL CE Loss: {self.avg_loss_mel_ce[-1][1]}\n"
        if self.avg_loss_text_ce:
            log_text += f"Current Text CE Loss: {self.avg_loss_text_ce[-1][1]}\n"
        
        log_text += "===========================\n"
        return log_text

    def _parse_trainer_metrics(self, log_data):
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

        current_epoch = 0
        current_step = None

        # Iterate over each line in the log
        for line in log_data.split('\n'):
            line = ansi_escape.sub('', line)

            # Parse avg losses per epoch
            if "avg_loss:" in line:
                value = float(line.split(":")[1].split()[0].strip())
                self.avg_loss.append((current_epoch, value))

            if "avg_loss_text_ce:" in line:
                value = float(line.split(":")[1].split()[0].strip())
                self.avg_loss_text_ce.append((current_epoch, value))

            if "avg_loss_mel_ce:" in line:
                value = float(line.split(":")[1].split()[0].strip())
                self.avg_loss_mel_ce.append((current_epoch, value))

            # Parse step metrics
            if "loss:" in line and "avg_" not in line:
                value = float(line.split(":")[1].split()[0].strip())
                self.loss.append((current_step, value))

            if "loss_text_ce:" in line and "avg_" not in line:
                value = float(line.split(":")[1].split()[0].strip())
                self.loss_text_ce.append((current_step, value))

            if "loss_mel_ce:" in line and "avg_" not in line:
                value = float(line.split(":")[1].split()[0].strip())
                self.loss_mel_ce.append((current_step, value))

            # Parse learning rate
            if "current_lr:" in line:
                value = float(line.split(":")[1].strip())
                self.learning_rates.append((current_step, value))

            # Parse gradient norms
            if "grad_norm:" in line:
                value = float(line.split(":")[1].split()[0].strip())
                if not hasattr(self, 'grad_norms'):
                    self.grad_norms = []
                self.grad_norms.append((current_step, value))

            # Parse step and loader times
            if "step_time:" in line:
                value = float(line.split(":")[1].split()[0].strip())
                if not hasattr(self, 'step_times'):
                    self.step_times = []
                self.step_times.append((current_step, value))

            if "loader_time:" in line:
                value = float(line.split(":")[1].split()[0].strip())
                if not hasattr(self, 'loader_times'):
                    self.loader_times = []
                self.loader_times.append((current_step, value))

            # Parse training and validation loss per epoch
            if "training_loss:" in line:
                value = float(line.split(":")[1].strip())
                if not hasattr(self, 'training_losses'):
                    self.training_losses = []
                self.training_losses.append((current_epoch, value))

            if "validation_loss:" in line:
                value = float(line.split(":")[1].strip())
                if not hasattr(self, 'validation_losses'):
                    self.validation_losses = []
                self.validation_losses.append((current_epoch, value))

            # Update epoch and step trackers
            if "EPOCH:" in line:
                current_epoch = int(line.split('/')[0].split(':')[-1].strip())

            if "GLOBAL_STEP:" in line:
                current_step = int(line.split("GLOBAL_STEP:")[1].split()[0])

    def plot_metrics(self):
        # Get the directory of the current script
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        # Define the full path for the output PNG
        output_path = os.path.join(current_script_dir, "training_metrics.png")
        # Ensure log data is parsed before plotting
        if self.current_model_path:
            log_file = os.path.join(self.current_model_path, "trainer_0_log.txt")
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        log_data = f.read()
                        # print(f"[DEBUG] Successfully read {len(log_data)} characters from log file.")
                        # Explicitly call the parsing method to process log data
                        self._parse_trainer_metrics(log_data)
                except Exception as e:
                    print(f"[DEBUG] Error reading trainer log: {e}")
                    return output_path  # Return early if there's an issue reading the file
            else:
                print("[DEBUG] Log file not found. No data to plot.")
                return output_path  # Return early if log file is missing
        plt.rcParams.update({'font.size': 12})
        plt.style.use('dark_background')

        fig, axes = plt.subplots(3, 2, figsize=(24, 18))
        (ax1, ax2), (ax3, ax4), (ax5, ax6) = axes

        # Common styling for all axes
        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('black')
            for spine in ax.spines.values():
                spine.set_color('white')

        # 1. Epoch Metrics (looks good but add y-axis adjustment)
        if self.avg_loss or self.avg_loss_text_ce or self.avg_loss_mel_ce:
            ax1.set_title('Epoch Metrics (Avg Losses)')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss Value')
            if self.avg_loss:
                epochs, values = zip(*self.avg_loss)
                ax1.plot(epochs, values, color='green', linestyle='-', marker='o', markersize=8, label='Avg Loss')
            if self.avg_loss_text_ce:
                epochs, values = zip(*self.avg_loss_text_ce)
                ax1.plot(epochs, values, color='red', linestyle='-', marker='o', markersize=8, label='Avg Loss Text CE')
            if self.avg_loss_mel_ce:
                epochs, values = zip(*self.avg_loss_mel_ce)
                ax1.plot(epochs, values, color='blue', linestyle='-', marker='o', markersize=8, label='Avg Loss MEL CE')
            ax1.set_xlim(left=0)
            ax1.legend()

        # 2. Step-wise Loss Metrics (fix x-axis)
        if self.loss or self.loss_text_ce or self.loss_mel_ce:
            ax2.set_title('Step-wise Loss Metrics')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Loss Value')
            if self.loss:
                steps, values = zip(*self.loss)
                ax2.plot(steps, values, color='green', linestyle='-', marker='o', markersize=8, label='Loss')
            if self.loss_text_ce:
                steps, values = zip(*self.loss_text_ce)
                ax2.plot(steps, values, color='red', linestyle='-', marker='o', markersize=8, label='Loss Text CE')
            if self.loss_mel_ce:
                steps, values = zip(*self.loss_mel_ce)
                ax2.plot(steps, values, color='blue', linestyle='-', marker='o', markersize=8, label='Loss MEL CE')
            ax2.set_xlim(left=0)
            ax2.legend()

        # 3. Learning Rate Schedule (fix x-axis and scale)
        if self.learning_rates:
            ax3.set_title('Learning Rate Schedule')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Learning Rate Value')
            steps, values = zip(*self.learning_rates)
            ax3.plot(steps, values, color='yellow', linestyle='-', marker='o', markersize=8, label='Learning Rate')
            ax3.set_xlim(left=0)
            ax3.legend()

        # 4. Gradient Norm
        if hasattr(self, 'grad_norms') and self.grad_norms:
            ax4.set_title('Gradient Norm over Steps')
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Gradient Norm')
            ax4.plot(*zip(*self.grad_norms), color='purple', linestyle='-', marker='o', markersize=8, label='Gradient Norm')
            ax4.legend()

        # 5. Step and Loader Times
        if hasattr(self, 'step_times') and self.step_times:
            ax5.set_title('Step and Loader Times')
            ax5.set_xlabel('Step')
            ax5.set_ylabel('Time (seconds)')
            ax5.plot(*zip(*self.step_times), color='orange', linestyle='-', marker='o', markersize=8, label='Step Time')
            if hasattr(self, 'loader_times') and self.loader_times:
                ax5.plot(*zip(*self.loader_times), color='cyan', linestyle='-', marker='o', markersize=8, label='Loader Time')
            ax5.legend()

        # 6. Training vs Validation Loss
        if hasattr(self, 'training_losses') and self.training_losses:
            ax6.set_title('Training vs Validation Loss')
            ax6.set_xlabel('Epoch')
            ax6.set_ylabel('Loss Value')
            ax6.plot(*zip(*self.training_losses), color='green', linestyle='-', marker='o', markersize=8, label='Training Loss')
            if hasattr(self, 'validation_losses') and self.validation_losses:
                ax6.plot(*zip(*self.validation_losses), color='red', linestyle='-', marker='o', markersize=8, label='Validation Loss')
            ax6.legend()

        plt.tight_layout()
        for ax in axes.flat:
            ax.set_xlim(left=0)  # Ensure no negative x values
            ax.grid(True, alpha=0.3)  # Consistent grid

        # Save the file
        plt.savefig(output_path, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # Return the full path to the calling script
        return output_path