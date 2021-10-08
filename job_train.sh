#!/bin/bash
#SBATCH --job-name=training
#SBATCH --gres=gpu:2              # Number of GPUs (per node)
#SBATCH --mem=85G               # memory (per node)
#SBATCH --time=0-12:00            # time (DD-HH:MM)

###########cluster information above this line

###load environment 
module load anaconda/3
module load cuda/10.1
source ../dpsa/bin/activate

filename=train.sh
chmod +x $filename
cat $filename | tr -d '\r' > $filename.new && rm $filename && mv $filename.new $filename 

#sbatch . train.sh
. train.sh 