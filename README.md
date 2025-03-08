# Code for Reproducing our Results on VLN-CE-BERT.

1. Configure the environment and download the necessary files. 

First, navigate to the original [Sim2Sim](https://github.com/jacobkrantz/Sim2Sim-VLNCE) repository and follow steps 1-6 to create a conda environment and install the required data and model files. You will need to clone and install `habitat-lab` and `transformers` manually. We use local navigation policies, so step 2 can be skipped.

**Note:** If you have trouble configuring the conda environment, refer to `ENV_SETUP.md`, where we provide our exact steps for initialization and solutions for problems we encountered.

2. Install `thop`:

```bash
$ pip install thop
```
Then, paste the profile function we have provided in the standard VLN codebase into `thop/profile.py`.

3. Run our efficient VLN

To run our efficient VLN agent on the validation (unseen) set from R2R-CE, run the following command:

```bash
python run.py \
    --exp-config sim2sim_vlnce/config/sgm-local_policy_efficient.yaml \
    EVAL_CKPT_PATH_DIR ./data/models/RecVLNBERT-ce_vision-tuned.pth \
    LOG_FILE ./logs/testing.log
```