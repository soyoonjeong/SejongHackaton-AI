import pickle

with open("/tmp/err_execute_model_input_20241226-171448.pkl", "rb") as f:
    failed_input = pickle.load(f)
    print(failed_input)
