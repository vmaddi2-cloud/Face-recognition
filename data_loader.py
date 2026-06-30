"""
data_loader.py
---------------
Loads the face dataset, removes exact-duplicate images (the dataset ships
"- Copy" duplicates), converts every image to grayscale, resizes it to a
fixed resolution, and flattens it into a column vector so that the full
database can be stacked into a (m*n) x p matrix as required by the
PCA / Eigenface formulation in the assignment.
"""
import os
import hashlib
import numpy as np
import cv2

IMG_SIZE = (64, 64)  # (n_cols, n_rows) used for cv2.resize


def _file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_dataset(dataset_dir, img_size=IMG_SIZE, verbose=True):
    """
    Walks dataset_dir/<person_name>/*.jpg, dedupes identical files,
    converts to grayscale, resizes, flattens.

    Returns
    -------
    Face_Db : ndarray, shape (m*n, p)   -- each column is one face image
    labels  : ndarray, shape (p,)       -- integer class label per column
    label_names : list[str]             -- label index -> person name
    """
    people = sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    )

    columns = []
    labels = []
    label_names = people

    for label_idx, person in enumerate(people):
        person_dir = os.path.join(dataset_dir, person)
        seen_hashes = set()
        n_kept, n_dupe = 0, 0
        for fname in sorted(os.listdir(person_dir)):
            fpath = os.path.join(person_dir, fname)
            if not os.path.isfile(fpath):
                continue
            h = _file_hash(fpath)
            if h in seen_hashes:
                n_dupe += 1
                continue
            seen_hashes.add(h)

            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
            columns.append(img.flatten().astype(np.float64))
            labels.append(label_idx)
            n_kept += 1

        if verbose:
            print(f"  {person:10s}: kept {n_kept:3d} images, skipped {n_dupe:3d} duplicates")

    Face_Db = np.stack(columns, axis=1)  # (m*n, p)
    labels = np.array(labels)
    return Face_Db, labels, label_names


if __name__ == "__main__":
    Face_Db, labels, names = load_dataset("dataset/faces")
    print("Face_Db shape:", Face_Db.shape)
    print("Classes:", names)
    print("Images per class:", np.bincount(labels))
