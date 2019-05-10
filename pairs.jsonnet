{
    "train_data_path": "/projects/instr/19sp/cse481n/GatesNLP/pairs_train.txt",
    "validation_data_path": "/projects/instr/19sp/cse481n/GatesNLP/pairs_dev.txt",

    "dataset_reader": {
      "type": "pairs_reader",
      "token_indexers": {
        "tokens": {
          "type": "single_id"
        }
      }
    },
    "model": {
        "type": "relevance_classifier",
        "text_field_embedder": {
              "token_embedders": {
                "tokens": {
                  "type": "embedding",
                  "embedding_dim": 40
                }
              }
            },
            "encoder": {
              "type": "rnn",
              "bidirectional": false,
              "input_size": 40,
              "hidden_size": 80,
              "num_layers": 1
            },
            "classifier_feedforward": {
              "input_dim": 80,
              "num_layers": 1,
              "hidden_dims": [2],
              "activations": ["linear"],
              "dropout": [0.2]
        }
    },
    "iterator": {
        "type": "basic",
        "batch_size": 64
    },
    "trainer": {
        "optimizer": "adam",
        "num_serialized_models_to_keep": 2,
        "num_epochs": 40,
        "patience": 10,
        "cuda_device": 1
    }

}