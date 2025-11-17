import os

# directories to skip entirely
SKIP_DIRS = {"data", "results", "experiments", "src"}

def tree(dir_path, prefix=""):
    contents = sorted(os.listdir(dir_path))
    pointers = ["├── "] * (len(contents) - 1) + ["└── "]

    for pointer, name in zip(pointers, contents):
        path = os.path.join(dir_path, name)

        # skip unwanted directories
        if name in SKIP_DIRS and os.path.isdir(path):
            print(prefix + pointer + name + "  [skipped]")
            continue

        print(prefix + pointer + name)

        if os.path.isdir(path):
            extension = "│   " if pointer == "├── " else "    "
            tree(path, prefix + extension)

if __name__ == "__main__":
    tree(".")
