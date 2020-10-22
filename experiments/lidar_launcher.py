from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def common_options(self):
        return [
            # Command 0
            Options(
                dataroot="./datasets/hdl32_to_pandar64",
                name="hdl32_to_pandar64_CUT",
                CUT_mode="CUT",
                gpu_ids=0
            ),

            # Command 1
            Options(
                dataroot="./datasets/hdl32_to_pandar64",
                name="hdl32_to_pandar64_FastCUT",
                CUT_mode="FastCUT",
                gpu_ids=0
            )
        ]

    def commands(self):
        return ["python train.py " + str(opt) for opt in self.common_options()]

    def test_commands(self):
        # RussianBlue -> Grumpy Cats dataset does not have test split.
        # Therefore, let's set the test split to be the "train" set.
        return ["python test.py " + str(opt.set(phase='train')) for opt in self.common_options()]
