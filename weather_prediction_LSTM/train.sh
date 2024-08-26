python main.py \
    --csv_path "seattle-weather.csv" \
    --batch_size 4 \
    --epochs 100 \
    --lr 0.001 \
    --output_dir "output/n1h1s1_epoch100" \
    --num_layers 1 \
    --hidden_size 1 \
    --sequence_length 1

    #--do_eval # test


# bash train.sh