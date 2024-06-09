
python extract_ofa_net.py -n ofa_mbv3_d234_e346_k357_w1.0 -ns 1000 -g 5 -log gpu_export_profiling.txt --deploy --backend_config_path $AUTOTAILOR_HOME/configs/backends/samsung0.json --command_config_path $AUTOTAILOR_HOME/configs/commands/ncnn_cpu_f0_fp16.json
python extract_ofa_net.py -n ofa_mbv3_d234_e346_k357_w1.0 -ns 1000 -g 5 --add_register -log gpu_register_profiling.txt
python extract_ofa_net.py -n ofa_mbv3_d234_e346_k357_w1.0 -ns 1000 -log cpu_profiling.txt
python extract_ofa_net.py -n ofa_mbv3_d234_e346_k357_w1.0 -ns 1000 --add_register -log cpu_register_profiling.txt
python extract_ofa_blocks.py $AUTOTAILOR_HOME/configs/backends/samsung0.json $AUTOTAILOR_HOME/configs/commands/ncnn_cpu_f0_fp16.json -n ofa_mbv3_d234_e346_k357_w1.0 -ns 1000 -g 5 -log block_export_profiling.txt