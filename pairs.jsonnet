{
    "train_data_path": "/projects/instr/19sp/cse481n/GatesNLP/supervised_pairs/pairs_train.txt",
    "validation_data_path": "/projects/instr/19sp/cse481n/GatesNLP/supervised_pairs/pairs_dev.txt",

    "dataset_reader": {
      "type": "pairs_reader",
      "token_indexers": {
        "tokens": {
          "type": "single_id"
        },
        "token_characters": {
          "type": "characters",
          "min_padding_length": 5
        }
      }
    },
    "model": {
        "type": "relevance_classifier",
        "text_field_embedder": {
          "token_embedders": {
            "tokens": {
              "type": "embedding",
              "embedding_dim": 50
            },
            "token_characters": {
              "type": "character_encoding",
              "embedding": {
                "embedding_dim": 8
              },
              "encoder": {
                "type": "cnn",
                "embedding_dim": 8,
                "num_filters": 50,
                "ngram_filter_sizes": [
                  5
                ]
              },
              "dropout": 0.2
            },
            "elmo": {
              "type": "elmo_token_embedder",
              "options_file": "/cse/web/courses/cse447/19wi/assignments/resources/elmo/elmo_small_options.json",
              "weight_file": "/cse/web/courses/cse447/19wi/assignments/resources/elmo/elmo_small_weights.hdf5",
              "do_layer_norm": false,
              "dropout": 0.5
            }
          }
        },
        "encoder": {
          "type": "lstm",
          "input_size": 356,
          "hidden_size": 100,
          "num_layers": 2,
          "dropout": 0.4,
          "bidirectional": true
        },
        "classifier_feedforward": {
          "input_dim": 200,
          "num_layers": 2,
          "hidden_dims": [100, 2],
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
        "cuda_device": 0,
        "validation_metric": "+accuracy"
    }

}