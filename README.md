# Instructions

## Dewarp Video

sbatch dewarp.s BiochemS1/Session_2_1400_Sensor_8/Eight.mp4 BiochemS1/Session_2_1400_Sensor_8/EightDewarped.mp4

## Extract Poses

sbatch postureOpenPose.s /scratch/xao1/TDCP/smaller.mp4 /scratch/xao1/TDCP/postures.avi /scratch/xao1/TDCP/postures/



## Correct Posture Video

sbatch correctPostureVideo.s \
    /scratch/xao1/BiochemS1/Session_1_0930_Sensor_3/ThreePostures.avi \
    /scratch/xao1/BiochemS1/Session_1_0930_Sensor_3/ThreePostures.mp4

## Extract Audio

sbatch extractAudio.s /scratch/xao1/TDCP/smaller.mp4 

## Normalize Audio

sbatch normalizeAudio.s TDCP/smaller.wav TDCP/normalized.wav


## Create Transcript

sbatch createTranscript.s /scratch/xao1/TDCP/normalized.wav /scratch/xao1/TDCP 5



## Segmenting People

sbatch segmentingPeople.s /scratch/xao1/TDCP/smaller.mp4 /scratch/xao1/TDCP/


srun --gres=gpu:1 --time=00:20:00 --pty /bin/bash
srun --time=00:20:00 --pty /bin/bash

singularity exec --nv \
      --overlay /scratch/work/public/singularity/openpose1.7.0-cuda11.1.1-cudnn8-devel-ubuntu20.04-dep.sqf:ro \
	    /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash /home/xao1/Code/CollaborationAnalysis/extractPoses.sh $VIDEOFILE $OUTFILE $OUTJSON

singularity exec --nv \
	    /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash /home/xao1/Code/CollaborationAnalysis/extractPoses.sh $VIDEOFILE $OUTFILE $OUTJSON

srun --gres=gpu:1 --mem=8GB --time=01:00:00 --pty /bin/bash

SINGULARITY_IMAGE=/scratch/work/public/singularity/ubuntu-20.04.1.sif
OVERLAY_FILE=/scratch/work/public/examples/greene-getting-started/overlay-15GB-500K-pytorch.ext3
singularity exec --nv --overlay $OVERLAY_FILE $SINGULARITY_IMAGE /bin/bash

source /ext3/miniconda3/bin/activate

conda activate /scratch/xao1/asr/nemo

python examples/asr/transcribe_speech.py \
 pretrained_name="stt_en_conformer_transducer_large" \
 audio_dir="/scratch/xao1/NeMo/audio"


singularity exec \
      --overlay /scratch/work/public/singularity/openpose1.7.0-cuda11.1.1-cudnn8-devel-ubuntu20.04-dep.sqf:ro \
	    /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash 

ffmpeg -y -i FourDewarped.mp4 -vf normalize=blackpt=black:whitept=white:smoothing=0 FourNormalized.mp4

VIDEOFILE=/scratch/xao1/
/home/xao1/Code/CollaborationAnalysis/extractPoses.sh $VIDEOFILE $OUTFILE $OUTJSON


Session_1_0930_Sensor_3
Session_1_0930_Sensor_4
Session_1_1100_Sensor_2
Session_1_1100_Sensor_5
Session_1_1100_Sensor_6
Session_1_1100_Sensor_7
Session_1_1100_Sensor_9
Session_1_1230_Sensor_3
Session_1_1230_Sensor_4
Session_1_1230_Sensor_5
Session_1_1230_Sensor_8
Session_1_1400_Sensor_1
Session_1_1400_Sensor_3
Session_1_1400_Sensor_5
Session_1_1400_Sensor_6
Session_1_1400_Sensor_7
Session_1_1400_Sensor_9

Session_2_0930_Sensor_1
Session_2_0930_Sensor_2
Session_2_0930_Sensor_3
Session_2_1100_Sensor_5
Session_2_1100_Sensor_6
Session_2_1100_Sensor_7
Session_2_1100_Sensor_8
Session_2_1230_Sensor_2
Session_2_1230_Sensor_3
Session_2_1230_Sensor_4
Session_2_1230_Sensor_9
Session_2_1400_Sensor_1
Session_2_1400_Sensor_2
Session_2_1400_Sensor_5
Session_2_1400_Sensor_6
Session_2_1400_Sensor_7
Session_2_1400_Sensor_8