---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:5933
- loss:TripletLoss
widget:
- source_sentence: There is an inverse correlation between Patient age and success
    rates.
  sentences:
  - Oh! So close to retirement.
  - Hes in excellent health. This was his first hospitalisation since breaking his
    leg at 23. Or 22, Im not sure anymore.
  - He was in the Navy not the Marines.
- source_sentence: get her consent. Shes moved on! new hub, new kid. She wants nothing
    to do with Drews death. Or me.
  sentences:
  - Now, hold on! Hold on! Oh yeah, I said Rachels name, but it didnt mean anything,
    Okay? Shesshes just a friend and thats all! Thats all!
  - Youre angry because your kid died. More than that, because you dont have an answer.
    People need answers.
  - Why did Gillick give me ketamine during my surgery
- source_sentence: Im ordering her cancer treatment to be continued. Why does it cost
    $2,300 to fix a coffee maParkne?
  sentences:
  - Yeah, yeah, yeah, save it, were busy. Luke, give us another half hour with your
    mom. We need to do some tests. Nice kid. Take her off the psych meds,
  - Because, II shouldve called! I threw her at his man nipples!
  - Chemo worked because cells are basically tumors. Chemo shrunk them. Youre still
    gonna say no, arent you
- source_sentence: This one works in financial district. She can get tips, give you
    leg up in market. What is fudgey Gonzalez?
  sentences:
  - Bosley. Either tell him hes an idiot, or tell me why Im wrong.
  - Pam! You cant be serious.
  - Uh, imagine a vanilla Gonzalez, but from the other side.
- source_sentence: Does this have anything to do with Addie?
  sentences:
  - Lets say yes.
  - Check it out, no one will tell me where Emily is, so Im gonna send 72 longstem,
    red roses to Emilys parents house, one for
  - Sure. and having them sitting in my office schmoozing about their favourite Algerian
    surfing movies, thats a much better system. Wait a sec. Were you in Row D
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy
model-index:
- name: SentenceTransformer
  results:
  - task:
      type: triplet
      name: Triplet
    dataset:
      name: dev evaluator
      type: dev_evaluator
    metrics:
    - type: cosine_accuracy
      value: 0.5451482534408569
      name: Cosine Accuracy
  - task:
      type: triplet
      name: Triplet
    dataset:
      name: final evaluator
      type: final_evaluator
    metrics:
    - type: cosine_accuracy
      value: 0.8827493190765381
      name: Cosine Accuracy
---

# SentenceTransformer

This is a [sentence-transformers](https://www.SBERT.net) model trained. It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
- **Maximum Sequence Length:** 128 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False}) with Transformer model: RobertaModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Does this have anything to do with Addie?',
    'Lets say yes.',
    'Check it out, no one will tell me where Emily is, so Im gonna send 72 longstem, red roses to Emilys parents house, one for',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Triplet

* Datasets: `dev_evaluator` and `final_evaluator`
* Evaluated with [<code>TripletEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.TripletEvaluator)

| Metric              | dev_evaluator | final_evaluator |
|:--------------------|:--------------|:----------------|
| **cosine_accuracy** | **0.5451**    | **0.8827**      |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 5,933 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>sentence_2</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                        | sentence_2                                                                       |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            | string                                                                           |
  | details | <ul><li>min: 3 tokens</li><li>mean: 13.46 tokens</li><li>max: 34 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 19.83 tokens</li><li>max: 51 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 19.0 tokens</li><li>max: 50 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                | sentence_1                                                                                                                                                     | sentence_2                                                                                                                  |
  |:--------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------|
  | <code>specifically told you not to assume . Can we at least assume that Im not dying tomorrow? Whereas this kid...</code> | <code>PET rEveals sEveral more hotspots. But theyre nonspecific...</code>                                                                                      | <code>Well, I did mention the Mars Rover incident to that FBI agent and probably cost Howard his security clearance.</code> |
  | <code>How can we do that if we know youre not?</code>                                                                     | <code>You dont know anything! Except, hopefully, our Patient on anticonvulsive medication has a seizure.</code>                                                | <code>Now come on. Well, Im glad we worked things out.</code>                                                               |
  | <code>Why? No way youre just doing her a favour.</code>                                                                   | <code>ER is standing room only. Which means Camerons bound to make a mistake. Find it so I can blackmail her. As far as you know, this is way more than</code> | <code>You know what you should do? Take a vacation.</code>                                                                  |
* Loss: [<code>TripletLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#tripletloss) with these parameters:
  ```json
  {
      "distance_metric": "TripletDistanceMetric.EUCLIDEAN",
      "triplet_margin": 5
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `num_train_epochs`: 8
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 8
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 8
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | Training Loss | dev_evaluator_cosine_accuracy | final_evaluator_cosine_accuracy |
|:------:|:----:|:-------------:|:-----------------------------:|:-------------------------------:|
| -1     | -1   | -             | 0.5451                        | -                               |
| 0.6739 | 500  | 3.4522        | -                             | -                               |
| 1.3477 | 1000 | 1.8387        | -                             | -                               |
| 2.0216 | 1500 | 1.5216        | -                             | -                               |
| 2.6954 | 2000 | 1.0493        | -                             | -                               |
| 3.3693 | 2500 | 0.8555        | -                             | -                               |
| 4.0431 | 3000 | 0.7493        | -                             | -                               |
| 4.7170 | 3500 | 0.5685        | -                             | -                               |
| 5.3908 | 4000 | 0.503         | -                             | -                               |
| 6.0647 | 4500 | 0.3924        | -                             | -                               |
| 6.7385 | 5000 | 0.3252        | -                             | -                               |
| 7.4124 | 5500 | 0.29          | -                             | -                               |
| -1     | -1   | -             | -                             | 0.8827                          |
| 0.6739 | 500  | 0.3696        | -                             | -                               |
| 1.3477 | 1000 | 0.4362        | -                             | -                               |
| 2.0216 | 1500 | 0.3908        | -                             | -                               |
| 2.6954 | 2000 | 0.2616        | -                             | -                               |
| 3.3693 | 2500 | 0.2105        | -                             | -                               |
| 4.0431 | 3000 | 0.1877        | -                             | -                               |
| 4.7170 | 3500 | 0.1406        | -                             | -                               |
| 5.3908 | 4000 | 0.1141        | -                             | -                               |
| 6.0647 | 4500 | 0.1136        | -                             | -                               |
| 6.7385 | 5000 | 0.0708        | -                             | -                               |
| 7.4124 | 5500 | 0.0638        | -                             | -                               |


### Framework Versions
- Python: 3.11.11
- Sentence Transformers: 3.4.1
- Transformers: 4.48.3
- PyTorch: 2.5.1+cu124
- Accelerate: 1.3.0
- Datasets: 3.3.2
- Tokenizers: 0.21.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### TripletLoss
```bibtex
@misc{hermans2017defense,
    title={In Defense of the Triplet Loss for Person Re-Identification},
    author={Alexander Hermans and Lucas Beyer and Bastian Leibe},
    year={2017},
    eprint={1703.07737},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->