import pickle


def dump_to_pickle(data, file_path: str):
    try:
        with open(file_path, "wb") as file:
            pickle.dump(data, file)
        print(f"Data successfully dumped to {file_path}")
    except Exception as e:
        print(f"An error occurred while dumping data: {e}")


def load_from_pickle(file_path: str):
    try:
        with open(file_path, "rb") as file:
            data = pickle.load(file)
        print(f"Data successfully loaded from {file_path}")
        return data
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return None
