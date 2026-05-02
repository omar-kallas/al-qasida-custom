CUDA_VISIBLE_DEVICES=0 ./run_baseline_1.sh 2>&1 | while IFS= read -r line; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"
done > run_baseline_1.log &
PID1=$!

CUDA_VISIBLE_DEVICES=1 ./run_baseline_2.sh 2>&1 | while IFS= read -r line; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"
done > run_baseline_2.log &
PID2=$!

wait $PID1 $PID2

runpodctl stop pod 5ue7kwfj72nv0g
