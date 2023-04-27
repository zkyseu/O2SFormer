CONFIG=$1
CHECKPOINT=$2

python test.py $CONFIG $CHECKPOINT --eval bbox
