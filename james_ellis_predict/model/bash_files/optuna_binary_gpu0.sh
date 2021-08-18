for i in 1 2 3
do
  python optunasearch.py "baseline+wws_binary+syn_binary-${i}" "model_data/baseline+wws+syn/data_binary.csv" "${i}" "0"
done
