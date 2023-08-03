import pickle

def load_pickle_file(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

def main():
    directories = ["daheng_6mm", "daheng_6mm_old", "daheng_12mm", "daheng_12mm_old", "daheng_25mm"]
    file_name = "calibration_params.pickle"  # replace this with your pickle file's name
    for directory in directories:
        print(f"\nCurrent file: {directory}")
        content = load_pickle_file("cam_calibration/cameras/" + directory + "/" + file_name)
        print("ret: ", content["ret"])
        print("mtx: ", content["mtx"])
        print("dist: ", content["dist"])
        print("w x h: ", content["calib_w"], " x ", content["calib_h"])

if __name__ == "__main__":
    main()
