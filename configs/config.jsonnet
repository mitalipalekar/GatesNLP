{
    "dataset_reader": {
        "type": "gatesnlp_dataset_reader",
        "max_sequence_length": 200,
        "token_indexers": {
            "elmo": {
                "type": "elmo_characters"
            },
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        }
    },
    "iterator": {
        "type": "basic",
        "batch_size": 32
    },
    "model": {
        "type": "mt_classifier",
        "classifier_feedforward": {
            "activations": [
                "relu",
                "linear"
            ],
            "dropout": [
                0.3,
                0
            ],
            "hidden_dims": [
                200,
                2
            ],
            "input_dim": 300,
            "num_layers": 2
        },
        "encoder": {
            "type": "rnn",
            "bidirectional": true,
            "hidden_size": 150,
            "input_size": 812,
            "num_layers": 2
        },
        "text_field_embedder": {
            "elmo": {
                "type": "elmo_token_embedder",
                "do_layer_norm": false,
                "dropout": 0.5,
                "options_file": "/cse/web/courses/cse447/19wi/assignments/resources/elmo/elmo_medium_options.json",
                "weight_file": "/cse/web/courses/cse447/19wi/assignments/resources/elmo/elmo_medium_weights.hdf5"
            },
            "tokens": {
                "type": "embedding",
                "embedding_dim": 300,
                "pretrained_file": "/cse/web/courses/cse447/19wi/assignments/resources/word2vec/GoogleNews-vectors-negative300.txt.gz",
                "trainable": true
            }
        }
    },
    "train_data_path": "src/extended_dataset.txt",
    "validation_data_path": "src/extended_dataset.txt",
    "trainer": {
        "cuda_device": 0,
        "num_epochs": 7,
        "optimizer": {
            "type": "adagrad"
        },
        "patience": 20,
        "validation_metric": "+accuracy"
    }
}