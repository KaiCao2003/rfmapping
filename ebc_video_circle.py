"""Cylinder recording entry: explicit config → shared eight-ray EBC overlay."""

from Utils.ebc_workflow import prepare_cylinder_video, video_main


def main(argv=None):
    return video_main(prepare_cylinder_video, argv)


if __name__ == "__main__":
    main()
