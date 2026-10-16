"""Basler entry: saved analysis → eight native samples → camera overlay."""

from EBB.ebc_workflow import prepare_rectangle_video, video_main


def main(argv=None):
    return video_main(prepare_rectangle_video, argv)


if __name__ == "__main__":
    main()
