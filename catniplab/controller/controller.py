"""
수학적이지 않고 프로그래밍에서만 필요한 함수들을 정의합니다.
"""


def mymax(itervar, **kwargs):
    """
    max인데, max의 결과가 일정 기준 이하면 default를 리턴
    :param dict itervar:
    :param kwargs:
    :return:
    """
    if 'key' in kwargs:
        mval = max(itervar, key=kwargs['key'])
    else:
        mval = max(itervar)
    default = kwargs['default'] if 'default' in kwargs else ValueError
    if 'underbound' in kwargs:
        mval = default if itervar[mval] < kwargs['underbound'] else mval
    return mval


def move_keyvalue(srcdict: dict, desdict: dict, key):
    """
    기존 dict의 key와 value을 목표 dict에 추가하고 기존 dict의 key와 value를 삭제합니다.
    어떤 의미에서는 cut and paste라고 볼 수 있습니다.

    :param dict srcdict: source dictionary
    :param dict desdict: destination dictionary
    :param key: key for source and destination dictionary
    :return: Nothing
    """
    if key in srcdict:
        desdict[key] = srcdict.pop(key)
    return


def classionary(content, target):
    """
    content와 target으로 나눠진 learning data를 target에 따라 묶은 dict로 만들어준다.

    :param list content: array-like. it contains content for learning
    :param list target: array-like. it contains target for learning
    :return dict:
    """
    kv = {}

    for c, t in zip(content, target):
        if t in kv:
            kv[t].append(c)
        else:
            kv[t] = [c]

    return kv


def folding_160311(content, target):
    """
    16년 03월 11일에 작성한 folding algorithm

    - 클래스별로 나눠서 학습시킨다.

    - unknown을 정해서 일부 class를 unknown으로 설정하며 fold한다.

    - 0%, 10%, 20%, 30%, 40%, 50%로 나눈다.\n
      ex. 10%: unknown: 10, 20, 30, 40, 50 나머지는 known

    - 그런데 빡세게(?) 코딩했는데 애초에 python의 for문은 yield되서 순서가 무작워...ㄷㄷ
      따라서 크게 의미는 없음...ㅋㅋ

    :param list content: array-like. it contains content for learning
    :param list target: array-like. it contains target for learning
    :return:
        content for learing,
        target for learning,
        content for examinating,
        target for examinating
    """
    size = 6
    csize = size - 1
    lcon, ltar = [[] for _ in range(size)], [[] for _ in range(size)]
    econ, etar = [[] for _ in range(size)], [[] for _ in range(size)]

    learningmatters = classionary(content, target)

    for i, k in enumerate(learningmatters):
        modi = i % 10
        if modi >= csize:
            for _i in range(modi - csize + 1):
                ltar[_i].append(k), lcon[_i].extend(learningmatters[k])
        else:
            for _i in range(size):
                ltar[_i].append(k), lcon[_i].extend(learningmatters[k])
        # testing은 모든 class의 data를 다 넣습니다.
        for _i in range(size):
            etar[_i].append(k), econ[_i].extend(learningmatters[k])

    return lcon, ltar, econ, etar
