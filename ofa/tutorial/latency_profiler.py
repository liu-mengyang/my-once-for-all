import os
import sys
import json
import time

AUTOTAILOR_HOME = os.getenv('AUTOTAILOR_HOME')
sys.path.append(AUTOTAILOR_HOME)

from infra.connector.connector import BackendConnector
from infra.converter.converter import Converter


class LatencyProfiler(object):
    def __init__(self, backend_path, command_path):
        self.command_info = json.load(open(command_path))
        self.connector = BackendConnector(backend_path)
        self.converter = Converter()
        self.convert_total_time = 0
        self.send_total_time = 0
        self.profile_total_time = 0
        self.counter = 0
        self.st = None
        
    def predict_efficiency(self, model, input_shape):
        if self.st is None:
            self.st = time.time()
        
        self.counter += 1
        pred = None
        
        convert_time = 0
        send_time = 0
        profile_time = 0
        
        tik = time.time()
        self.converter.torch2tflite(model.cpu(),
                    model_name='ofa_profile',
                    data_shape=input_shape,
                    verbose=False,
                    save_dir='.')
        tok = time.time()
        convert_time = tok-tik
        
        base_dir = 'ofa_profile'
        tflite_path = base_dir + '_float32.tflite'
        tik = time.time()
        self.connector.send_model(tflite_path)
        tok = time.time()
        send_time = tok-tik
        
        model_name = 'ofa_profile_float32.tflite'
        model_path = os.path.join(self.connector.model_dir, model_name)
        configs = self.command_info
        
        self.command_info["HW0"] = input_shape[2]
        self.command_info["HW1"] = input_shape[3]
        self.command_info["CIN"] = input_shape[1]
        
        configs['shape'] = f'[{input_shape[2]},{input_shape[3]},{input_shape[1]}]'
        
        tik = time.time()
        res = self.connector.profile(model_path=model_path, configs=configs, enable_latency_constraint=False, verbose=True)
        tok = time.time()
        profile_time = tok-tik
        
        self.convert_total_time += convert_time
        self.send_total_time += send_time
        self.profile_total_time += profile_time
        if 'Segmentation fault' in res:
            return 1000
        elif 'Timeout' in res:
            return 1000
        elif 'Notgreen' in res:
            return 1000
        elif 'ANEURALNETWORKS_OP_FAILED' in res:
            return 1000
        else:
            latency = self.connector.parse(res)['latency']
        print(f"profile latency {latency}")
        
        pred = float(str(latency).split(' +- ')[0])
        
        self.ed = time.time()
        
        print("Count: "+str(self.counter))
        print("Convert: "+str(self.convert_total_time))
        print("Send: "+str(self.send_total_time))
        print("Profile: "+str(self.profile_total_time))
        print("Total time: "+str(self.ed-self.st))
        
        return pred