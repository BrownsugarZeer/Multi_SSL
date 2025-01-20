import argparse


def parsing_params():
    parser = argparse.ArgumentParser(description="Run a SRP-PHAT")
    parser.add_argument(
        "-s",
        "--src",
        type=int,
        default=None,
        help="how many sources we want to detect.",
    )
    parser.add_argument(
        "-c", "--channels", type=int, default=6, help="microphone output channels."
    )
    parser.add_argument(
        "-i",
        "--ignored_channels",
        type=list,
        default=[0, 5],
        help="which channel should be disable.",
    )
    parser.add_argument(
        "-w", "--wave", type=str, default="", help="a path of wave file."
    )
    args = parser.parse_args()
    return args
