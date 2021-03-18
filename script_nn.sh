execute(){
    for d in $3; do
        for model in $4; do
            for batch_size in $1; do
                for lr in $2; do
                    python3.6 nn.py "valid" ${batch_size} ${lr} ${d} ${model}
                done
            done
        done
    done
}

execute_test(){
    for d in $1; do
        for model in $2; do
            for fold in $(seq 0 4); do
                python3.6 nn.py "test" ${fold} ${d} ${model}
            done
        done
    done
}

#execute '8 16 32 64' '0.001 0.01 0.1 1'
#execute '1' '0.001 0.01 0.1 1' 'debate' 'attention_tfidf'
#execute '3' '0.1'
#execute '1' '0.1' 'sentistrength_bbc' 'attention_glove_tfidf'
#execute '4' '0.01 0.1 1' 'sentistrength_rw' 'attention_glove_tfidf'
#execute '1' '0.1' 'pang_movie' 'attention_glove_tfidf'

#execute_test 'debate' 'attention_tfidf'
#execute_test 'debate' 'attention_glove'
#execute_test 'debate' 'attention_glove_tfidf'
#execute_test 'sentistrength_rw' 'attention_glove_tfidf'
execute_test 'pang_movie' 'attention_glove_tfidf'
