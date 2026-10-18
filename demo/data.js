window.RECAP = {
  "examples": {
    "audio": {
      "id": "1-101404-A-34.wav",
      "truth": "can opening",
      "source": "esc50/audio/1-101404-A-34.wav",
      "asset": "assets/can-opening.wav",
      "selection": "Lexicographically first item ID among the 2,000 shared clips for which, for each of Gemma 2, Jina v5 Omni-small, and Omni Mini 0.9B, the inner-selected native prompt is wrong and the nested logistic reader is correct for all three fixed outer seeds. Omni Mini 2.3B is displayed on the same clip as a post-selection context model and did not participate in the selection rule.",
      "models": [
        {
          "key": "gemma2",
          "name": "EmbeddingGemma 2",
          "predictions": [
            {
              "native_prediction": "clapping",
              "reader_prediction": "can opening",
              "seed": 20261011
            },
            {
              "native_prediction": "clapping",
              "reader_prediction": "can opening",
              "seed": 20261012
            },
            {
              "native_prediction": "clapping",
              "reader_prediction": "can opening",
              "seed": 20261013
            }
          ]
        },
        {
          "key": "jina",
          "name": "Jina v5 Omni-small",
          "predictions": [
            {
              "native_prediction": "sneezing",
              "reader_prediction": "can opening",
              "seed": 20261011
            },
            {
              "native_prediction": "sneezing",
              "reader_prediction": "can opening",
              "seed": 20261012
            },
            {
              "native_prediction": "sneezing",
              "reader_prediction": "can opening",
              "seed": 20261013
            }
          ]
        },
        {
          "key": "omni23",
          "name": "Omni Mini 2.3B",
          "predictions": [
            {
              "native_prediction": "glass breaking",
              "reader_prediction": "can opening",
              "seed": 20261011
            },
            {
              "native_prediction": "glass breaking",
              "reader_prediction": "can opening",
              "seed": 20261012
            },
            {
              "native_prediction": "glass breaking",
              "reader_prediction": "can opening",
              "seed": 20261013
            }
          ]
        },
        {
          "key": "omni09",
          "name": "Omni Mini 0.9B",
          "predictions": [
            {
              "native_prediction": "crackling fire",
              "reader_prediction": "can opening",
              "seed": 20261011
            },
            {
              "native_prediction": "crackling fire",
              "reader_prediction": "can opening",
              "seed": 20261012
            },
            {
              "native_prediction": "crackling fire",
              "reader_prediction": "can opening",
              "seed": 20261013
            }
          ]
        }
      ]
    },
    "transfer": [
      {
        "id": "AVE_test_--I2373YpgM",
        "truth": "Horse",
        "target": "video",
        "model": "Omni Mini 2.3B",
        "matched": "Horse",
        "cross": "Male speech, man speaking",
        "asset": "assets/horse.mp4",
        "source": {
          "audio": "ave/test/audio/AVE_test_--I2373YpgM.wav",
          "frame_pattern": "ave/test/frames/AVE_test_--I2373YpgM/frame_{index:02d}.jpg",
          "frame_numbers": [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15
          ]
        },
        "selection": "lexicographically first item_id satisfying the direction-specific exact argmax pattern"
      },
      {
        "id": "AVE_test_-44qKBR8bGQ",
        "truth": "Acoustic guitar",
        "target": "audio",
        "model": "Omni Mini 2.3B",
        "matched": "Acoustic guitar",
        "cross": "Banjo",
        "asset": "assets/guitar.wav",
        "source": {
          "audio": "ave/test/audio/AVE_test_-44qKBR8bGQ.wav",
          "frame_pattern": "ave/test/frames/AVE_test_-44qKBR8bGQ/frame_{index:02d}.jpg",
          "frame_numbers": [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15
          ]
        },
        "selection": "lexicographically first item_id satisfying the direction-specific exact argmax pattern"
      }
    ],
    "images": [
      {
        "id": "ObjectNet:638",
        "truth": "cellphone",
        "prompt": "bath towel",
        "ridge": "cellphone",
        "predictions": [
          {
            "seed": 20261011,
            "native": "bath towel",
            "ridge": "cellphone"
          },
          {
            "seed": 20261012,
            "native": "bath towel",
            "ridge": "cellphone"
          },
          {
            "seed": 20261013,
            "native": "bath towel",
            "ridge": "cellphone"
          }
        ],
        "asset": "assets/cellphone.jpg",
        "source": "mmeb/ObjectNet/image_s0003472.jpg",
        "dataset": "ObjectNet",
        "model": "Jina",
        "kind": "rescue",
        "selection": "Visually selected from 156 held-out ObjectNet images where selected prompts miss and nested Ridge succeeds across all three seeds. Chosen to make the target object and the incorrect background-related label easy to see; not the first qualifying item."
      },
      {
        "truth": "bath towel",
        "prompt": "bath towel",
        "ridge": "dishrag or hand towel",
        "asset": "assets/bath-towel.jpg",
        "source": "mmeb/ObjectNet/image_s0006091.jpg",
        "dataset": "ObjectNet",
        "model": "Jina",
        "kind": "regression",
        "id": "ObjectNet:30",
        "selection": "First manifest-ordered ObjectNet native-correct / nested-Ridge-wrong example agreeing across all three seeds."
      }
    ],
    "note": "Outcome-selected held-out illustrations. Scores come from saved evaluations; this demo performs no model inference."
  },
  "width": {
    "native_prompt_ba_minimum": 0.11088221663081142,
    "native_prompt_ba_native": 0.6510086539532118,
    "ridge_alpha10_ba_minimum": 0.22507534196343598,
    "ridge_alpha10_ba_native": 0.5969126605240556,
    "gap_interaction_min_minus_native": 0.16828911876178088,
    "minimum_dimension": 32,
    "native_dimension": 1024,
    "seed_gaps": [
      {
        "gap_native": -0.051003417323039035,
        "gap_minimum": 0.11935521257825807,
        "gap_interaction_min_minus_native": 0.1703586299012971
      },
      {
        "gap_native": -0.06465437489326087,
        "gap_minimum": 0.10258386364671494,
        "gap_interaction_min_minus_native": 0.16723823853997583
      },
      {
        "gap_native": -0.04663018807116903,
        "gap_minimum": 0.12064029977290065,
        "gap_interaction_min_minus_native": 0.16727048784406967
      }
    ]
  },
  "audio_scores": {
    "gemma2": {
      "native_prompt_selected_inner": 0.2368333333333333,
      "ridge_nested": 0.8096666666666666,
      "logistic_nested": 0.8401666666666667
    },
    "jina": {
      "native_prompt_selected_inner": 0.7371666666666666,
      "ridge_nested": 0.9298333333333333,
      "logistic_nested": 0.9341666666666667
    },
    "ovis": {
      "native_prompt_selected_inner": 0.8375000000000001,
      "ridge_nested": 0.9440000000000001,
      "logistic_nested": 0.9453333333333334
    },
    "omni23": {
      "native_prompt_selected_inner": 0.401,
      "ridge_nested": 0.8491666666666666,
      "logistic_nested": 0.8541666666666666
    },
    "omni09": {
      "native_prompt_selected_inner": 0.6030000000000001,
      "ridge_nested": 0.8986666666666667,
      "logistic_nested": 0.8985000000000001
    }
  },
  "transfer": {
    "name": "Omni-Embed-Mini 2.3B",
    "dimension": 2048,
    "balanced_accuracy_matrix": [
      [
        0.6297401332327802,
        0.36661571679954036
      ],
      [
        0.3254262281835811,
        0.9504610177404295
      ]
    ]
  },
  "budget": {
    "best_feasible_native_only": 0.6809833501019398,
    "selector": 0.9072004587967383
  }
};
