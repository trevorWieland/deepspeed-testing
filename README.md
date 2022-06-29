# deepspeed-testing

Each benchmark was run on a range of up to 4 RTX 3070s using [runpod.io](runpod.io),
pytorch 1.12, CUDA 11.3, and deepspeed v0.6.5.

## Installation

For now, the only supported installation method is through runpod.io, but the docker image is
publicly available. You may also clone this repo and build/modify the containers yourself, if
you need a different version of pytorch/cuda.

### RunPod
For use in RunPod, first create an account and load up some money at [runpod.io](runpod.io).
Once you're ready to deploy, create a new template in the `Templates` tab under `MANAGE`.
It's easiest to duplicate the RunPod Pytorch template that's already there.

Change the Template name to whatever you like, then change the `Container Image` to `trevorwieland/deepspeed:runpod`.
You do not need any registry credentials or  extra docker commands.

After setting the Disk sizes to your needs (defaults were okay for this benchmark), make sure that
`Expose TCP Port` is set to 22, and Volume Mount Path is `/workspace`.

Then open up `Environment Variables`. We only need one environment variable, which will be `PUBLIC_KEY`.
This should be a generated public key for accessing over ssh, which RunPod will walk you through creating if
you go to settings. You can also just google it.

Once all of this is done, be sure to save the template! Then head to Deploy on either of
`SECURE CLOUD` or `COMMUNITY CLOUD`, configure your platform of choice, select your new template,
and launch the pod.

This pod will only be available to access over ssh, and doesn't have jupyter lab installed, so click on
the `connect` button on your created pod in the `My Pods` page to see how to connect to your pod over ssh.

If you're having difficulty, make sure your pod is actually running. If you chose `spot` pricing, it
might have been shut down due to someone else outbidding you!

### Local Docker install
If you're here, you probably know what you're doing!

Just know that you'll need nvidia docker support in order to push gpus to a docker container,
which is only available on linux. The dockerhub repo for this project is at [trevorwieland/deepspeed](https://hub.docker.com/r/trevorwieland/deepspeed).
Currently the only available tag is `:runpod`, but if there is a usecase for other tags we will add them.

You can also build your own by cloning this repo and modifying the Dockerfile and start script to suit
your needs!

## Translation Benchmarking
This section is most relevant for the sugoi translation enhancement project, but for the
sake of this benchmark, the translation is instead EN->VI translation using the iwslt2015-en-vi
dataset on huggingface. This is because this dataset was readily available, relatively small size,
and easy to use.

### Results

| Run Kind          | #GPU | Runtime | Samples/Second |
|-------------------|------|---------|----------------|
| Direct            | 4    | 1654.14 | 80.6           |
| Torch Distributed | 1    |         |                |
| Torch Distributed | 2    | 1263.44 | 105.5          |
| Torch Distributed | 3    | 881.25  | 151.3          |
| Torch Distributed | 4    | 712.35  | 187.2          |
| Deepspeed Zero2   | 1    |         |                |
| Deepspeed Zero2   | 2    |         |                |
| Deepspeed Zero2   | 3    |         |                |
| Deepspeed Zero2   | 4    |         |                |
| Deepspeed Zero3   | 1    |         |                |
| Deepspeed Zero3   | 2    |         |                |
| Deepspeed Zero3   | 3    |         |                |
| Deepspeed Zero3   | 4    |         |                |

### Commands
If using the docker container, make sure the following commands are run from the `workspace` folder.

The command to simply run all gpus at once with no distributed scheduler is as follows:

    python transformers/examples/pytorch/translation/run_translation.py \
    --model_name_or_path t5-small --per_device_train_batch_size 8 \
    --output_dir deepspeed-testing/examples/language-modeling/output/ \
    --overwrite_output_dir --fp16 --do_train --num_train_epochs 1 \
    --dataset_name mt_eng_vietnamese --dataset_config "iwslt2015-en-vi" \
    --source_lang en --target_lang vi --source_prefix "translate English to Vietnamese: "

The command to run using distributed scheduling on a set `NUM_GPU` is as follows:

    python -m torch.distributed.launch --nproc_per_node={NUM_GPU} \
    transformers/examples/pytorch/translation/run_translation.py \
    --model_name_or_path t5-small --per_device_train_batch_size 8 \
    --output_dir deepspeed-testing/examples/language-modeling/output/ \
    --overwrite_output_dir --fp16 --do_train --num_train_epochs 1 \
    --dataset_name mt_eng_vietnamese --dataset_config "iwslt2015-en-vi" \
    --source_lang en --target_lang vi --source_prefix "translate English to Vietnamese: "

The command to run using deepspeed on a set `NUM_GPU` is as follows:

    deepspeed --num_gpus={NUM_GPU} transformers/examples/pytorch/translation/run_translation.py \
    --deepspeed transformers/tests/deepspeed/ds_config_zero2.json \
    --model_name_or_path t5-small --per_device_train_batch_size 8 \
    --output_dir deepspeed-testing/examples/language-modeling/output/ \
    --overwrite_output_dir --fp16 --do_train --num_train_epochs 1 \
    --dataset_name mt_eng_vietnamese --dataset_config "iwslt2015-en-vi" \
    --source_lang en --target_lang vi --source_prefix "translate English to Vietnamese: "

For deepspeed, both the built in zero2 and zero3 files were used for the benchmark, though a custom
optimized file would be great to have to see if it can improve further.

The batch size was kept constant across all three runs, however with some additional tinkering with deepspeed,
the batch size should be able to increase, leading to better performance.

## CLM Benchmarking

Test deepspeed's capabilities in doing CLM, because thats what the main tutorial had it doing.
Unfortunately, there seems to be an issue with deepspeed and gpt models, as I have not been
able to get it working a single time due to a `__flops__` [attribute missing error](https://github.com/microsoft/DeepSpeed/issues/2046).
(This wasn't posted by me, but I'm having the same issue)

Directly using huggingface training, as well as torch distributed has worked however, so I'll include
the time for those tests to run.

### Results

| Run Kind          | #GPU | Runtime | Samples/Second |
|-------------------|------|---------|----------------|
| Direct            | 4    | 58.8959 | 39.358         |
| Torch Distributed | 1    | 41.2717 | 56.164         |
| Torch Distributed | 2    | 20.867  | 111.085        |
| Torch Distributed | 3    | 16.8644 | 137.449        |
| Torch Distributed | 4    | 13.659  | 169.69         |
| Deepspeed Zero2   | 1    | N/A     | N/A            |
| Deepspeed Zero2   | 2    | N/A     | N/A            |
| Deepspeed Zero2   | 3    | N/A     | N/A            |
| Deepspeed Zero2   | 4    | N/A     | N/A            |
| Deepspeed Zero3   | 1    | N/A     | N/A            |
| Deepspeed Zero3   | 2    | N/A     | N/A            |
| Deepspeed Zero3   | 3    | N/A     | N/A            |
| Deepspeed Zero3   | 4    | N/A     | N/A            |

### Commands
If using the docker container, make sure the following commands are run from the `workspace` folder.

The command to simply run all gpus at once with no distributed scheduler is as follows:

    python transformers/examples/pytorch/translation/run_translation.py \
    --model_name_or_path t5-small --per_device_train_batch_size 8 \
    --output_dir deepspeed-testing/examples/language-modeling/output/ \
    --overwrite_output_dir --fp16 --do_train --num_train_epochs 1 \
    --dataset_name mt_eng_vietnamese --dataset_config "iwslt2015-en-vi" \
    --source_lang en --target_lang vi --source_prefix "translate English to Vietnamese: "

The command to run using distributed scheduling on a set `NUM_GPU` is as follows:

    python -m torch.distributed.launch --nproc_per_node={NUM_GPU} \
    transformers/examples/pytorch/translation/run_translation.py \
    --model_name_or_path t5-small --per_device_train_batch_size 8 \
    --output_dir deepspeed-testing/examples/language-modeling/output/ \
    --overwrite_output_dir --fp16 --do_train --num_train_epochs 1 \
    --dataset_name mt_eng_vietnamese --dataset_config "iwslt2015-en-vi" \
    --source_lang en --target_lang vi --source_prefix "translate English to Vietnamese: "
