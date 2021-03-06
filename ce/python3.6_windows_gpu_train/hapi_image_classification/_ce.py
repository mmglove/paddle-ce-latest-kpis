####this file is only used for continuous evaluation test!
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.append(os.environ['ceroot'])
from kpi import CostKpi, DurationKpi, AccKpi

#### NOTE kpi.py should shared in models in some way!!!!
resnet50_train_loss_kpi = CostKpi('resnet50_train_loss', 0.1, 0, actived=True, desc='train cost')
resnet50_eval_loss_kpi = CostKpi('resnet50_eval_loss', 0.1, 0, actived=True, desc='eval cost')

mobilenet_v1_train_loss_kpi = CostKpi('mobilenet_v1_train_loss', 0.1, 0, actived=True, desc='train cost')
mobilenet_v1_eval_loss_kpi = CostKpi('mobilenet_v1_eval_loss', 0.1, 0, actived=True, desc='eval cost')

mobilenet_v2_train_loss_kpi = CostKpi('mobilenet_v2_train_loss', 0.1, 0, actived=True, desc='train cost')
mobilenet_v2_eval_loss_kpi = CostKpi('mobilenet_v2_eval_loss', 0.1, 0, actived=True, desc='eval cost')

vgg16_train_loss_kpi = CostKpi('vgg16_train_loss', 0.1, 0, actived=True, desc='train cost')
vgg16_eval_loss_kpi = CostKpi('vgg16_eval_loss', 0.1, 0, actived=True, desc='eval cost')


tracking_kpis = [ resnet50_train_loss_kpi, resnet50_eval_loss_kpi, mobilenet_v1_train_loss_kpi, mobilenet_v1_eval_loss_kpi, mobilenet_v2_train_loss_kpi, mobilenet_v2_eval_loss_kpi, vgg16_train_loss_kpi, vgg16_eval_loss_kpi]

def parse_log(log):
    '''
    This method should be implemented by model developers.

    The suggestion:

    each line in the log should be key, value, for example:

    "
    
    "
    '''
    for line in log.split('\n'):
        fs = line.strip().split('\t')
        print(fs)
        if len(fs) == 3 and fs[0] == 'kpis':
            print("-----%s" % fs)
            kpi_name = fs[1]
            kpi_value = float(fs[2])
            yield kpi_name, kpi_value


def log_to_ce(log):
    kpi_tracker = {}
    for kpi in tracking_kpis:
        kpi_tracker[kpi.name] = kpi

    for (kpi_name, kpi_value) in parse_log(log):
        print(kpi_name, kpi_value)
        kpi_tracker[kpi_name].add_record(kpi_value)
        kpi_tracker[kpi_name].persist()


if __name__ == '__main__':
    log = sys.stdin.read()
    print("*****")
    print(log)
    print("****")
    log_to_ce(log)
