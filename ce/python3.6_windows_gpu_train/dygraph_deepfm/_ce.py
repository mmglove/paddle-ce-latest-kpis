# this file is only used for continuous evaluation test!

import os
import sys

sys.path.append(os.environ['ceroot'])
from kpi import CostKpi
from kpi import DurationKpi
from kpi import AccKpi

# NOTE kpi.py should shared in models in some way!!!!
train_duration_kpi = DurationKpi('train_duration', 0.08, 0, actived=True)
test_auc_kpi = AccKpi('test_auc', 0.08, 0, actived=True, desc='test_auc')


tracking_kpis = [train_duration_kpi,
                 test_auc_kpi]


def parse_log(log):
    '''
    This method should be implemented by model developers.

    The suggestion:

    each line in the log should be key, value, for example:

    "
    kpis\ttest_auc\t
    kpis\ttest_times\t
    "
    '''
    for line in log.split('\n'):
        fs = line.strip().split('\t')
        print(fs)
        if len(fs) == 3 and fs[0] == 'kpis':
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
    log_to_ce(log)
