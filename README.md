# event_cnn_minimal
Minimal code for loading models trained for ECCV'20

Run with 

    python -m event_cnn_minimal.load_model -c /path/to/model/checkpoint.pth -s /path/to/try/and/save/jit
    
The script will load the model and then attempt to save it to a TorchScript object (this line is currently commented, since it causes things to bug out, to try and save with `jit` uncomment the very last line).
