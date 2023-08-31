from glob import glob


def dataset_check(base_dir: str = "./brain4cars_data/"):
    for fold_num in range(5):
        train_file_list = [x.split("/")[-1] for x in glob(f"{base_dir}/train/fold{fold_num}/**/*.avi") if "road_vids" not in x]
        val_file_list = [x.split("/")[-1] for x in glob(f"{base_dir}/val/fold{fold_num}/**/*.avi") if "road_vids" not in x]
        print(f"Number of overlapping files for fold {fold_num}: {len(set(train_file_list) & set(val_file_list))}")

if __name__ == "__main__":
    dataset_check()
