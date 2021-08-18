for i in 1 2 
do
  python optunasearch.py "baseline+wws_closeness+syn_cbrank-${i}" "model_data/baseline+wws+syn/data_closeness_cbrank.csv" "${i}" "0"
done
