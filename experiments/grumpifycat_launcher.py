from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def common_options(self):
        return [
            # Command 0
            Options(
                dataroot="./datasets/grumpifycat",
                name="grumpifycat_CUT",
                CUT_mode="CUT"
            ),

            # Command 1--train FastCUT
            Options(
                dataroot="/mnt/ahp_urban/hist/CUT",
                name="grumpifycat_FastCUT_grayscale",
                CUT_mode="FastCUT",
                checkpoints_dir="checkpoints/",
                input_nc=1, # grayscale input
                output_nc=1 # grayscale output
                # continue_train="--continue_train"
                # gpu_ids='0,1,2,3'
            ),
            # Command 2 -- test FastCUT
            Options(
                dataroot="/mnt/ahp_urban/hist/CUT",
                results_dir="/mnt/ahp_urban/hist/CUT/FastCUT_historical_to_modern",
                name="grumpifycat_FastCUT_grayscale",
                CUT_mode="FastCUT",
                checkpoints_dir="checkpoints/",
                num_test=int(1e6), # when testing, use all images
                input_nc=1, # grayscale input
                output_nc=1 # grayscale output
                # continue_train="--continue_train"
                # gpu_ids='0,1,2,3'
            )
        ]

    def commands(self):
        return ["python train.py " + str(opt) for opt in self.common_options()]

    def test_commands(self):
        # RussianBlue -> Grumpy Cats dataset does not have test split.
        # Therefore, let's set the test split to be the "train" set.
        return ["python test.py " + str(opt.set(phase='train')) for opt in self.common_options()]
