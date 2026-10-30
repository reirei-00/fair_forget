import json
import random

from rmu.config import load_config, repo_path

random.seed(42)
STEREO_PATHS = [repo_path(path) for path in load_config()["data"]["stereoset_files"]]

def get_json_dict(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def get_stereoset_data(
    cat_type: str = "intrasentence",
    paths=STEREO_PATHS,
    batch_size: int = 4,
    split: bool = True,
):
    """Load StereoSet gender examples and return them in batches.

    Items are shuffled before they are divided into two groups. The forget
    batches contain stereotype sentences from the first group; the retain
    batches contain anti-stereotype sentences from the second group. When
    ``split`` is false, both sentence sets are taken from all gender items.

    Args:
        cat_type: StereoSet category to load (for example, ``intrasentence``).
        paths: JSON files whose ``data[cat_type]`` entries should be combined.
        batch_size: Maximum number of sentences in each returned batch. The
            final batch may contain fewer sentences.
        split: Whether to use separate groups for forgetting and retaining.

    Returns:
        A pair ``(forget_batches, retain_batches)``. Each value is a list of
        sentence batches, and each batch is a list of sentence dictionaries.

    Raises:
        ValueError: If ``batch_size`` is not a positive integer.
    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    full_stereo_list = []
    for path in paths:
        full_stereo_list.extend(get_json_dict(path)["data"][cat_type])

    gender_list = list(filter(lambda x: x["bias_type"] == "gender", full_stereo_list))
    random.shuffle(gender_list)
    n = len(gender_list)
    forget_list, retain_list = gender_list[:n//2], gender_list[n//2:]
    if not split:
        forget_list = gender_list
        retain_list = gender_list
    forget_data = []
    retain_data = []
    for fitem in forget_list:
        for sentence in fitem["sentences"]:
            if sentence["gold_label"] != "stereotype":
                continue
            forget_data.append(sentence) 

    for fitem in retain_list:
        for sentence in fitem["sentences"]:
            if sentence["gold_label"] != "anti-stereotype":
                continue
            retain_data.append(sentence)

    def batchify(data):
        return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    return batchify(forget_data), batchify(retain_data)

if __name__ == "__main__":
    forget_batches, retain_batches = get_stereoset_data()

    forget_count = sum(len(batch) for batch in forget_batches)
    retain_count = sum(len(batch) for batch in retain_batches)
    print(f"Forget: {forget_count} sentences in {len(forget_batches)} batches")
    print(f"Retain: {retain_count} sentences in {len(retain_batches)} batches")

    if forget_batches:
        print("First forget example:")
        print(forget_batches[0][0])
    if retain_batches:
        print("First retain example:")
        print(retain_batches[0][0])
