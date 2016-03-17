"""
local storage에 접근하는 io를 정의합니다.
"""
import os


path_com = "D:/Document/niplab/LIG"
path_ver = "/april"  # 어떤 데이터인지 알려주세요
# path_pjt =
path_naultech = 'D:/workspace/OneClass_detect/resource'


def naultech_learn():
    """
    :return: row가 class, column이 data인 double array입니다.
    """
    data, target = [], []
    with open(os.path.join(path_naultech, "Learn.txt"), 'r') as file:
        data, res = [], []

        for line in file.readlines():
            l = line.split(sep='   ')
            res.append([float(x) for x in l if x != ''])

        zipres = zip(*res)

        for t, _data in enumerate(zipres):
            for d in _data:
                data.append(d), target.append(t)

    return data, target


def naultech_test():
    """
    :return: row가 class, column이 data인 double array입니다.
    """
    data, target = [], []
    climax = 5000
    with open(os.path.join(path_naultech, "Learn.txt"), 'r') as file:
        data, res = [], []

        for line in file.readlines():
            l = line.split(sep='   ')
            res.append([float(x) for x in l if x != ''])

        for i, ptn in enumerate(res):
            t = int(i / 5000)
            data.append(ptn)
            target.append(t)

    return data, target
