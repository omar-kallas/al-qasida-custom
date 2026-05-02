CUDA_VISIBLE_DEVICES=0 ./run_ablation_1.sh 2>&1 | while IFS= read -r line; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"
done > ablation_second_run_1.log &
PID1=$!

CUDA_VISIBLE_DEVICES=1 ./run_ablation_2.sh 2>&1 | while IFS= read -r line; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"
done > ablation_second_run_2.log &
PID2=$!

wait $PID1 $PID2

runpodctl stop pod t48c83uu3g6tsi