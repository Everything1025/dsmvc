EOS = 1e-10
eps = 1e-8

DEFAULT_CONFIG = {
    "Trento": {
        "n_neighbors": 50,
        "lr":  1e-05,
        "nlayers": 4,
        "dropout": 0.5,
        "in_dim": 128,
        "hidden_dim": 256,
        "emb_dim": 512,
        "proj_dim": 512,
        "temperature": 1.0,
        "bias": 1e-05,
        "top_k_rate": 0.001,
        "min_neighbors": 5,
        "max_neighbors": 15,

        "w_smi": 1,
        "w_umi": 1,
        "w_dis": 0.1
    },
    "gpu_id": "1",
    "n_epoches": 50,
    "remove_bkg": True,
    "sparse": False,
    "refine": True,
    "log_path": "./log",
    "save_path": "./save",
    "result_path": "./result",
    "score_path": "./score"
}
