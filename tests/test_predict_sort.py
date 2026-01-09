from pathlib import Path

from sharp.cli.predict import _natural_sort_key


def test_natural_sort_key_orders_relative_paths() -> None:
    input_root = Path("input")
    paths = [
        Path("input/001/10.jpg"),
        Path("input/004/1.jpg"),
        Path("input/001/2.jpg"),
    ]

    sorted_paths = sorted(paths, key=lambda path: _natural_sort_key(path, input_root))

    assert [path.relative_to(input_root).as_posix() for path in sorted_paths] == [
        "001/2.jpg",
        "001/10.jpg",
        "004/1.jpg",
    ]
