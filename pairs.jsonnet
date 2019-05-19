{
    "train_data_path": "/projects/instr/19sp/cse481n/GatesNLP/supervised_pairs/quadruple_pairs_train.txt",
    "validation_data_path": "/projects/instr/19sp/cse481n/GatesNLP/supervised_pairs/quadruple_pairs_dev.txt",

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
              "pretrained_file": "/cse/web/courses/cse447/19wi/assignments/resources/glove/glove.840B.300d.txt.gz",
              "embedding_dim": 300,
              "trainable": false
            }
          }
        },
        "encoder": {
          "type": "lstm",
          "input_size": 300,
          "hidden_size": 50,
          "num_layers": 2,
          "dropout": 0.4,
          "bidirectional": true
        },
        "classifier_feedforward": {
          "input_dim": 100,
          "num_layers": 2,
          "hidden_dims": [50, 2],
          "activations": ["tanh","linear"],
          "dropout": [0.4, 0.4]
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
        "cuda_device": 0
    }

}