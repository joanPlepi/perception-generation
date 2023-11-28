#!/bin/bash
#
#SBATCH --job-name=fine_tune_ft5
#SBATCH --output=../logs/ft5.out
#SBATCH --error=../logs/ft5.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=joanplepi@gmail.com
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

source /home/plepi/anaconda3/etc/profile.d/conda.sh
conda activate perception
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/plepi/anaconda3/lib/

python ../src/fine_tune_custom.py \
--text_to_use='title' \
--persona_amount=20 \
--batch_size=8 \
--max_input_length=100 \
--max_target_length=256 \
--num_epochs=6 \
--dataset_size=-1 \
--encoder_mode='encoder_persona' \
--split='verdicts' \
--model_name='custom_flant5' \
--persona_separate='True' \
--only_persona