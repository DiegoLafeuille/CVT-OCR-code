import pickle
import os

def load_pickle_file(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

def main():
    base_path = "cam_calibration/cameras/"
    directories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    file_name = "calibration_params.pickle"  # replace this with your pickle file's name
    for directory in directories:
        print(f"\nCurrent file: {directory}")
        content = load_pickle_file(base_path + directory + "/" + file_name)
        print("ret: ", content["ret"])
        print("mtx: ", content["mtx"])
        print("dist: ", content["dist"])
        print("w x h: ", content["calib_w"], " x ", content["calib_h"])

if __name__ == "__main__":
    main()

