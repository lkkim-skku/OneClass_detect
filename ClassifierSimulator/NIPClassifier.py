__author__ = 'leesuk kim'
"""
Radar signal Categorizer
version 2.0
Designed by leesuk kim(aka. LK)
Lang.: Python
Lang. ver.: 3.4.3

"""
import math
import time

import numpy as np
from scipy.stats import beta as scibeta
import scipy.stats as scistats
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


class Sample:
    m = 'mahalanobis'
    c = 'euclidean'
    g = 'gaussian'
    p = 'epanechnikov'
    ml = 'mahalanobis_logscaled'
    cl = 'euclidean_logscaled'
    gl = 'gaussian_logscaled'
    pl = 'epanechnikov_logscaled'
    x = ''

    def __init__(self, raw):
        self.raw = raw
        self.samples = {}
        self.x = Sample.x

    def set(self, mean, std):
        self.samples[Sample.m] = distance_mahalanobis(self.raw, mean, std)
        self.samples[Sample.c] = distance_euclidean(self.raw, mean)
        self.samples[Sample.g] = kernel_gaussian(self.raw, mean, std)
        self.samples[Sample.p] = kernel_epanechnikov(self.raw, mean, std)
        self.samples[Sample.ml] = logscaling(self.samples[Sample.m])
        self.samples[Sample.cl] = logscaling(self.samples[Sample.c])
        self.samples[Sample.gl] = logscaling(self.samples[Sample.g])
        self.samples[Sample.pl] = logscaling(self.samples[Sample.p])
        return self

    def get(self, *args):
        """
        계산한 sample을 가져온다.
        len(args) == 0이면 지정된 algorithm의 sample을 가져온다.
        len(args) == 1이면 args[0]에 지정된 algorithm의 sample을 가져온다.
        """
        if len(args) == 0:
            return self.samples[self.x]
        else:
            if args[0] in self.samples:
                return self.samples[args[0]]

    def __repr__(self):
        return repr(self.samples[self.x])

    def __str__(self):
        return str(self.samples[self.x])

    def __bytes__(self):
        return bytes(self.samples[self.x])

    def __format__(self, format_spec):
        return format(self.samples[self.x], format_spec=format_spec)

    def __lt__(self, other):
        return self.samples[self.x] < type_sample(other)

    def __le__(self, other):
        return self.samples[self.x] <= type_sample(other)

    def __eq__(self, other):
        return self.samples[self.x] == type_sample(other)

    def __ne__(self, other):
        return self.samples[self.x] != type_sample(other)

    def __gt__(self, other):
        return self.samples[self.x] > type_sample(other)

    def __ge__(self, other):
        return self.samples[self.x] >= type_sample(other)

    def __hash__(self):
        return hash(self.samples[self.x])

    def __add__(self, other):
        return self.samples[self.x] + type_sample(other)

    def __sub__(self, other):
        return self.samples[self.x] - type_sample(other)

    def __mul__(self, other):
        return self.samples[self.x] * type_sample(other)

    def __truediv__(self, other):
        return self.samples[self.x] / type_sample(other)

    def __floordiv__(self, other):
        return self.samples[self.x] // type_sample(other)

    def __mod__(self, other):
        return self.samples[self.x] % type_sample(other)

    def __divmod__(self, other):
        return divmod(self.samples[self.x], type_sample(other))

    def __pow__(self, power, modulo=None):
        return pow(self.samples[self.x], power, modulo)

    def __lshift__(self, other):
        return self.samples[self.x] << type_sample(other)

    def __rshift__(self, other):
        return self.samples[self.x] >> type_sample(other)

    def __and__(self, other):
        return self.samples[self.x] & type_sample(other)

    def __xor__(self, other):
        return self.samples[self.x] ^ type_sample(other)

    def __or__(self, other):
        return self.samples[self.x] | type_sample(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __rfloordiv__(self, other):
        return self.__floordiv__(other)

    def __rmod__(self, other):
        return self.__mod__(other)

    def __rdivmod__(self, other):
        return self.__divmod__(other)

    def __rpow__(self, power, modulo=None):
        return self.__pow__(power, modulo)

    def __rlshift__(self, other):
        return self.__lshift__(other)

    def __rrshift__(self, other):
        return self.__rshift__(other)

    def __rand__(self, other):
        return self.__and__(other)

    def __rxor__(self, other):
        return self.__xor__(other)

    def __ror__(self, other):
        return self.__or__(other)

    def __iadd__(self, other):
        self.samples[self.x] += type_sample(other)
        return self

    def __isub__(self, other):
        self.samples[self.x] -= type_sample(other)
        return self

    def __imul__(self, other):
        self.samples[self.x] *= type_sample(other)
        return self

    def __itruediv__(self, other):
        self.samples[self.x] /= type_sample(other)
        return self

    def __ifloordiv__(self, other):
        self.samples[self.x] //= type_sample(other)
        return self

    def __imod__(self, other):
        self.samples[self.x] %= type_sample(other)
        return self

    def __ipow__(self, other):
        self.samples[self.x] **= type_sample(other)
        return self

    def __ilshift__(self, other):
        self.samples[self.x] <<= type_sample(other)
        return self

    def __irshift__(self, other):
        self.samples[self.x] >>= type_sample(other)
        return self

    def __iand__(self, other):
        self.samples[self.x] &= type_sample(other)
        return self

    def __ixor__(self, other):
        self.samples[self.x] ^= type_sample(other)
        return self

    def __ior__(self, other):
        self.samples[self.x] |= type_sample(other)
        return self

    def __neg__(self):
        return -self.samples[self.x]

    def __pos__(self):
        return +self.samples[self.x]

    def __abs__(self):
        return abs(self.samples[self.x])

    def __invert__(self):
        return ~self.samples[self.x]

    def __int__(self):
        return int(self.samples[self.x])

    def __float__(self):
        return float(self.samples[self.x])

    def __round__(self, n=None):
        return round(self.samples[self.x], ndigits=n)

    def __getitem__(self, item):
        return self.raw[item]


def type_sample(samp):
    return samp.samples[samp.x] if type(samp) == Sample else samp


class Statistic:
    """더 간단하게 만들 수도 있지만 polymorphism을 위해 여러 case를 합침"""
    def __init__(self):
        self.data = []
        self._stats_ = {}
        self._isscalar_ = False

    def measure(self, data):
        self.data = data
        stats, self._isscalar_ = Statistic.measure_mvs(data)
        self._stats_.update(stats)

    @staticmethod
    def measure_mvs(data):
        stats = {}
        if type(data[0]) == list or type(data[0]) == tuple:
            isscalar = False
            _trans_ = list(zip(*data))
            stats['mean'] = [np.mean(x) for x in _trans_]
            stats['var'] = [np.var(x, ddof=1) for x in _trans_]
            stats['std'] = [x ** 0.5 for x in stats['var']]
        else:
            isscalar = True
            stats['mean'] = np.mean(data)
            stats['var'] = np.var(data)
            stats['std'] = stats['var'] ** 0.5

        return stats, isscalar

    def __setitem__(self, key, value):
        self._stats_[key] = value

    def __getitem__(self, item):
        return self._stats_[item]


class BetaFunction(Statistic):
    """
    이 class는 Statistic을 extand합니다.
    왜냐고? 각종 통계수치를 필요로 하기 때문이지.
    """
    def __init__(self, name, centroid, opts):
        # beta_engines = {
        #     'scipy': self.betafit_numpy,
        #     'abourizk': self.abourizk,
        #     'trapezoidal': self.trapezoidal
        # }
        shape_engines = {
            'mm': self.moment_match
        }
        self.name = name
        # self.engine_beta = opts['beta']
        self.engine_shape = opts['bse']
        # self.__beta__ = beta_engines[self.engine_beta]
        self.__betashape__ = shape_engines[self.engine_shape]
        self.centroid = centroid
        self.alpha, self.beta = 0., 0.
        self.samples = []
        super().__init__()

    def kstest(self, x):
        s = Sample(x).set(self.centroid['mean'], self.centroid['std'])
        fss = featurescaling(s, x_min=self._stats_['min'], x_max=self._stats_['max'])
        p = scibeta.cdf(fss, self.alpha, self.beta)
        if Sample.x == Sample.m or Sample.x == Sample.c:
            p = 1. - p

        return 0 if math.isnan(p) else p

    def kernelize(self, x):
        s = Sample(x).set(self.centroid['mean'], self.centroid['std'])
        fss = featurescaling(s, x_min=self._stats_['min'], x_max=self._stats_['max'])
        if fss < 0:
            fss = 0
        return fss

    def samplify(self, data):
        return [Sample(sample).set(self.centroid['mean'], self.centroid['std']) for sample in data]

    def aprx_betashape(self, samples):
        """beta distribution을 추정"""
        # samples = [Sample(sample).set(self.centroid['mean'], self.centroid['std']) for sample in data]
        # 모든 data를 이 kernel을 통해 생성된 sample로 변환
        samp_fs, self['min'], self['max'] = featurescaling(samples)
        self.samples = samples
        # normalize with feature scaling
        super().measure(samp_fs)
        # 평균, 분산 등 각종 통계치 계산
        self.alpha, self.beta = self.__betashape__()

        # CDF start
        hist, bins = histogram(samp_fs, 100)
        k = 0.
        cdf_ = []
        samplan = len(self.data)
        for h in hist:
            k += h / samplan
            cdf_.append(k)
        # CDF END
        self['betaECDF'] = cdf_

        # self['betaECDF'] = [x for x in sorted(set(self.data))]

        # 통계치로 beta shape parameter 계산
        # self.alpha, self.beta, _, _ = scibeta.fit(self['betaECDF'], self.alpha, self.beta)
        # TODO beta fitting 안 했습니다.
        pval = 0.
#        unpack = fit_betaparam_search(self['betaECDF'], self.alpha, self.beta)
#         unpack = search_betaparam(self['betaECDF'], self.alpha, self.beta)
#        d, pval, self.alpha, self.beta = unpack[0], unpack[1], unpack[2], unpack[3]
        d, pval = scistats.kstest(self['betaECDF'], scibeta.cdf, args=(self.alpha, self.beta))
        self['p-value'], self['D'] = pval, d

        # CDF start
        # hist, bins = histogram(self['betaECDF'], 100)
        # k = 0.
        # cdf_ = []
        # samplan = len(self.data)
        # for h in hist:
        #     k += h / samplan
        #     cdf_.append(k)
        # CDF END
        # d, pval = scistats.kstest(cdf_, scibeta.cdf, args=(self.alpha, self.beta))
        # self['p-value'], self['D'] = pval, d
        # plotname = '[hist]nofit)beta_' + self.name
        # plot_lines(plotname, ("p=%1.8lf\t" % self['p-value']) + (r"$\alpha$=%1.8lf" % self.alpha) + "\t" + (r"$\beta$=%1.8lf" % self.beta),
        #            dict(name='Beta CDF', x=bins, y=scibeta.cdf(bins, self.alpha, self.beta)
        #                 ), dict(name='empirical CDF', x=bins, y=cdf_))
        return self

    def moment_match(self):
        """Moment Matching"""
        lower, upper = 0., 1.  # feature scaling한 뒤니까 당연히 0과 1이죠.
        ml = self['mean'] - lower
        um = upper - self['mean']
        a = (ml / (upper - lower)) * (((ml * um) / self['var']) - 1)
        b = a * (um / ml)
        return a, b
    #
    # @staticmethod
    # def betafit_numpy(alpha, beta, samples):
    #     """
    #     numpy에 있는 알고리즘으로 fitting
    #     """
    #     ba, bb, _, _ = beta.expect(samples, alpha, beta)
    #     bcdf = scibeta.cdf(samples, ba, bb)
    #     return bcdf
    #
    # @staticmethod
    # def abourizk(alpha, beta, samples):
    #     """
    #     Fitting Beta Distributions Based on Sample Data로 Fitting
    #     기본 알고리즘
    #     """
    #     bcdf = []
    #     return bcdf
    #
    # @staticmethod
    # def trapezoidal(alpha, beta, samples):
    #     """
    #     CPON에 나와있는 Fitting Algorithm
    #     """
    #     bcdf = []
    #     return bcdf

    pass


class Cluster(object):
    """
    Classic Singleton Pattern
    """
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Cluster, cls).__new__(cls)
        return cls.instance

    def __init__(self, opts: dict):
        self.engines = {
            'fld': self.fld,
            'lk': self.lk
        }
        if opts['cluster'] in self.engines:
            self.engine = self.engines[opts['cluster']]
        else:
            print("CLUSTER ENGINE DOES NOT EXIST.")
        pass

    def clustering(self, name, data):
        return self.engine(name, data)

    @staticmethod
    def fld(name, data):
        ctrd, iss = Statistic.measure_mvs(data)
        return [ctrd]

    @staticmethod
    def lk(name, data):
        ctrd, iss = Statistic.measure_mvs(data)
        return [ctrd]


class Class(Statistic):
    """
    1. Clustering
    2. Building Beta
    it knows KERNEL
    """
    def __init__(self, opts: dict, name, data):
        super().__init__()
        self.name, self.data, self.opts = name, data, opts
        centroids = Cluster(opts).clustering(name, data)
        # kernels = [Kernel(name, c, kernel=opts['kernel']) for c in centroids]
        self.alpha, self.beta = 0., 0.
        self.bfp = [BetaFunction(self.name, c, opts=self.opts) for c in centroids]

    def fit(self, totaldata):
        """
        class의 betafunction에 data를 입력하면 betafunctino은
        그 data를 kernel로 quantize해서 sample을 만든 다음
        그 sample들을 featurescaling하고
        featured sample들로 beta shape parameter를 만든 후
        kstest해서 pvalue 만들면 끝.
        """
        for bf in self.bfp:
            bf.aprx_betashape(bf.samplify(totaldata))

            # print("CLASS:" + self.name)
            # print("Mahalanobis\tEuclidean\tGaussian\tEpanechnikov\tlog(Mahalanobis)\tlog(Euclidean)\tlog(Gaussian)\tlog(Epanechnikov)")
            # for sample in bf.samples:
            #     print("%16.15lf\t%16.15lf\t%16.15lf\t%16.15lf\t%16.15lf\t%16.15lf\t%16.15lf\t%16.15lf" %
            #           (sample.get(Sample.m),sample.get(Sample.c),sample.get(Sample.g), sample.get(Sample.p),
            #            sample.get(Sample.ml),sample.get(Sample.cl),sample.get(Sample.gl), sample.get(Sample.pl)))

        print(self.name + "fit complete")
        return self

    def moment_match(self):
        """Moment Matching"""
        lower, upper = 0., 1.  # feature scaling한 뒤니까 당연히 0과 1이죠.
        ml = self['mean'] - lower
        um = upper - self['mean']
        a = (ml / (upper - lower)) * (((ml * um) / self['var']) - 1)
        b = a * (um / ml)
        return a, b

    def predict(self, x):
        """predict할 때 사용"""
        # 참고로 원래 그냥 sum하는게 아니라 weight를 곱해줘야 합니다.
        # 그런데 그 weight도 kernel에 붙이는거라 귀찮아서 그냥 넘김.
        pval = sum(bf.kstest(x) for bf in self.bfp)

        return pval


class CPON:
    """
    Class Probability Output Network
    """
    def fit(self, data, target):
        """fit"""
        self._classes_ = {}
        self.predicts = []
        self._data_ = []
        self._target_ = []
        self.pred_pval = []

        self._data_, self._target_ = data, target
        dataset = {}
        for d, t in zip(data, target):
            if t not in dataset:
                dataset[t] = []
            dataset[t].append(d)

        for t in dataset:
            self._classes_[t] = Class(opts=self.kwargs, name=t, data=dataset[t]).fit(dataset[t])

        if len(self._classes_) == 2:
            targets = list(self._classes_)
            opdist = {targets[0]: dict(name=targets[0], x=[]), targets[1]: dict(name=targets[1], x=[])}
            for name in self._classes_:
                outputs = [self.output(x) for x in self._classes_[name].data]
                preds = [x[name] for x in outputs]
                h, s = histogram(preds, 100)
                freqsum = sum(h)
                h = [x / freqsum for x in h]
                width = s[1] - s[0]
                if width < 1e-5:  # 너무 작으면 아얘 고정해서 보정
                    s = [x / 100 for x in range(100)]
                    width = 0.01
                opdist[name]['w'] = width / 2
                opdist[name]['y'], opdist[name]['x'] = h, s
            opdist[targets[1]]['x'] = [1. - x for x in opdist[targets[1]]['x']]  # '-' 클래스는 inverse
            plot_bars('od_' + targets[0] + targets[1], "OutputDist.-" + targets[0] + " and " + targets[1], opdist[targets[0]], opdist[targets[1]])

            opdist = {targets[0]: dict(name=targets[0], x=[], y=[]), targets[1]: dict(name=targets[1], x=[], y=[])}
            for name in self._classes_:
                outputs = [self.output(x) for x in self._classes_[name].data]
                for x in outputs:
                    for p in x:
                        if p == name:
                            opdist[p].get('x').append(x[p])
                            # opdist[p]['x'].append(x[p])
                        else:
                            opdist[p].get('y').append(x[p])
                            # opdist[p]['y'].append(x[p])
            temp = opdist[targets[1]]['y']
            opdist[targets[1]]['y'] = opdist[targets[1]]['x']
            opdist[targets[1]]['x'] = temp
            # TODO 2차원 그래프로 트레이닝 데이터의 분포를 나타냄
            # plot_dots('od2d' + targets[0] + targets[1], "OD2D-" + targets[0] + " and " + targets[1], opdist[targets[0]], opdist[targets[1]])

            pass

    def output(self, sample):
        """
        :param sample: one of data
        :return: p-value
        """
        """
        이걸 좀 더 카와이하게 바꿔보자고.
        일단 모든 Class에 대해서 kstest를 하면 각 clas마다 p-value가 나오겠지?
        그러면 그 p-value들을 모아서 '여.기.에.서. 각 class의 output을 만드는거야.
        이를 통해 각 class는 bfn을 가져야할 이유를 없앨 수 있지.
        게다가 연산속도는 n^n에서 n으로 줄어드니 이게 정답이지.
        """
        clspval = self.__predict__(sample)
        pval_sum = sum(clspval[k] for k in clspval)

        # pvalue_dict = {}
        # # TODO 아래의 코드는 에러가 있습니다
        # for k in clspval:
        #     k_neg = [x for x in clspval if k not in clspval]
        #     demoni = clspval[k] - k_neg + 1
        #     pvalue_dict[k] = clspval[k] / demoni
        # clspval = max(pvalue_dict, key=lambda x: pvalue_dict[x])

        if pval_sum != 1. and pval_sum != 0.:
            for k in clspval:
                clspval[k] /= pval_sum

        return clspval

    def __predict__(self, sample):
        return {k: self._classes_[k].predict(sample) for k in self._classes_}

    def predict(self, data):
        pred = []
        self.pred_pval = [{} for _ in data]
        for i, sample in enumerate(data):
            for name in self._classes_:
                self.pred_pval[i][name] = self._classes_[name].predict(sample)

            # for k in self._classes_:
            #     k_neg = [self.pred_pval[i][x] for x in self.pred_pval[i].keys() if k not in self.pred_pval[i].keys()]
            #     demoni = self.pred_pval[i][k] - sum(k_neg) + 1

        for pval in self.pred_pval:
            # predcls = max(pval, key=lambda x: pval[x], default="unknown")
            predcls = mymax(pval, key=lambda x: pval[x], default="unknown", underbound=0.05)
            pred.append(predcls)

        return pred

    def thread_predict(self, clsname, data):
        for i, sample in enumerate(data):
            p = self._classes_[clsname].predict(sample)
            self.pred_pval[i][clsname] = p

        print(clsname + ' predict complete')

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        if 'timestamp' not in kwargs:
            self.kwargs['timestamp'] = '%d' % int(time.time())
        if 'kernel' not in kwargs:
            self.kwargs['kernel'] = 'gaussian'
        if 'cluster' not in kwargs:
            self.kwargs['cluster'] = 'lk'
        if 'beta' not in kwargs:
            self.kwargs['beta'] = 'abourizk'
        if 'bse' not in kwargs:
            self.kwargs['bse'] = 'mm'
        if 'threadable' not in kwargs:
            self.kwargs['threadable'] = False
        if 'accaptance' not in kwargs:
            self.kwargs['accaptance'] = 0.05
        self._classes_ = {}
        self.predicts = []
        self._data_ = []
        self._target_ = []
        self.pred_pval = []
        Sample.x = Sample.g  # Sample type

    def __str__(self):
        tostring = "[" + self.__class__.__name__ + "]"
        for i, key in enumerate(self.kwargs):
            tostring += key + '=\''
            tostring += (self.kwargs[key] if type(self.kwargs[key]) == str else repr(self.kwargs[key])) + '\''
            tostring += ', ' if i < len(self.kwargs) - 1 else '\n'
        return tostring


def histogram(x: list, bins: int):
    """히스토그램"""
    x = [a for a in x]
    x.sort()
    xn, xx = x[0], x[-1]
    step = (xx - xn) / bins
    box, i = xn + step, 0
    histo = []
    for a in x:
        while a > box:
            histo.append(i)
            i = 0
            box += step
        i += 1
    if len(histo) == bins:
        histo[-1] += i
    else:
        # histo.append(i)
        while len(histo) < bins:
            histo.append(i)
            i = 0
    steps = []
    box = xn + step
    for _ in range(bins):
        steps.append(box)
        box += step
    return histo, steps


def formula_featurescaling(x, x_min, x_max):
    return 0 if x_max == x_min or x == x_min else (x - x_min) / (x_max - x_min)


def featurescaling(x, **kwargs):
    """
    kwargs가 없고 x가 list라면 featurescaling하면서 x의 min과 max를 리턴
    kwargs가 있으면 거기서 xmin과 xmax를 가져오면서 featurescaling
    """
    if 'x_min' in kwargs and 'x_max' in kwargs:
        return x if type(x) == str else formula_featurescaling(x, kwargs['x_min'], kwargs['x_max'])
    else:
        x_min, x_max = min(x), max(x)
        x_fs = [formula_featurescaling(a, x_min, x_max) for a in x]
        return x_fs, x_min, x_max


def formula_student(x, mean, std):
    return (x - mean) / std


def student(x, **kwargs):
    """Student's statistic"""
    if 'mean' in kwargs and 'std' in kwargs:
        return x if type(x) == str else formula_student(x, kwargs['mean'], kwargs['std'])
    else:
        m, s = np.mean(x), np.std(x, ddof=1)
        x_ss = [formula_student(a, m, s) for a in x]
        return x_ss, m, s


def logscaling(x):
    """log scaling"""
    return math.log(1 + x) / math.log(2)
    # 한빈이가 말한 logscaling...이거 이상함 ㅋㅋ
    # return math.log(-1 * math.log(x + 10e-20) + 10e-20)


def formula_gaussian_weight(d, weight):
    return math.exp(-1 * d * weight)


def kernel_weight(size):
    """
    weight generator for kernel
    :param size: length of range
    :return: weight
    """
    mid = int(size / 2)
    for i in range(size):
        yield 2 ** (i - mid)


def distance_euclidean(x, m):
    """
    Euclidean distance를 계산합니다.
    :param x: array-like, 거리를 계산할 대상
    :param m: array-like, 기준점(평균)
    :return: float-like
    """
    d = (sum((a - b) ** 2 for a, b in zip(x, m)) ** 0.5) / 12
    return d


def formular_mahalanobis(val, mean, std):
    return 0 if val == mean or std == 0 else ((val - mean) ** 2) / (std ** 2)


def distance_mahalanobis(x, mean, std):
    """
    Mahalanobis distance를 계산합니다.
    :param x: array-like, 거리를 계산할 대상
    :param mean: array-like, 평균
    :param std: array-like, 표준 편차
    :return: float-like
    """
    d = sum(formular_mahalanobis(a, m, s) for a, m, s in zip(x, mean, std)) / 12
    d **= 0.5
    return d


def formula_gaussian(mahalanobis):
    return math.exp(-0.5 * (mahalanobis ** 2))
    # return 1 / ((2 * math.pi) ** 0.5) * math.exp(-0.5 * (mahalanobis ** 2))  # ORIGINAL


def kernel_gaussian(iter_x, iter_m, iter_s):
    x = distance_mahalanobis(iter_x, iter_m, iter_s)
    x = formula_gaussian(x)
    return x


def kernel_epanechnikov(iter_x, iter_m, iter_s):
    """
    Epanechnokiv Kernel Function
    """
    x = distance_mahalanobis(iter_x, iter_m, iter_s)
    x = 1 - x ** 2 if abs(x) <= 1 else 0.
    # x = 0.75 * (1 - x ** 2) if abs(x) <= 1 else 0.  # ORIGINAL
    return x


def plot_lines(filename, title, *args):
    """
    그래프를 선으로 그립니다.
    :param filename: 파일명
    :param title: 그래프 제목
    :param args: 그래프를 그릴 자료
    이 자료는 dictionary이며, key로 'x', 'y', 'name'을 받습니다.
    :return:엄ㅋ슴ㅋ
    """
    fig, ax = plt.subplots()
    plt.title(title)
    for xy in args:
        x = xy['x']
        y = xy['y']
        plt.plot(x, y, label=xy['name'])
    plt.legend()
    fig.savefig(filename + '.png')
    plt.clf()
    pass


def plot_bars(filename, title, *args):
    """
    그래프를 bar chart 으로 그립니다.
    :param filename: 파일명
    :param title: 그래프 제목
    :param args: 그래프를 그릴 자료
    이 자료는 dictionary이며, key로 'x', 'y', 'name', 'w'를 받습니다.
    'w'는 bar의 너비입니다.
    :return:엄ㅋ슴ㅋ
    """
    fig, ax = plt.subplots()
    plt.title(title)
    palette = ['k', 'w', 'y', 'r']
    for xyw, color in zip(args, palette):
        x = xyw['x']
        y = xyw['y']
        w = xyw['w']
        plt.bar(x, y, width=w, color=color, label=xyw['name'])
    plt.xlim([-0.1, 1.1])
    # plt.legend()
    fig.savefig(filename + '.png')
    plt.clf()
    pass


def plot_dots(filename, title, *args):
    """
    그래프를 선으로 그립니다.
    :param filename: 파일명
    :param title: 그래프 제목
    :param args: 그래프를 그릴 자료
    이 자료는 dictionary이며, key로 'x', 'y', 'name'을 받습니다.
    :return:엄ㅋ슴ㅋ
    """
    fig, ax = plt.subplots()
    plt.title(title)
    palette = ['r.', 'b+']
    for xy, cl in zip(args, palette):
        x = xy['x']
        y = xy['y']
        plt.plot(x, y, cl, label=xy['name'])
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    fig.savefig(filename + '.png')
    plt.clf()
    pass


def mymax(itervar, **kwargs):
    """
    max인데, max의 결과가 일정 기준 이하면 default를 리턴
    :param itervar:
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


def search_betaparam(empirical_cdf, alpha, beta):
    a, b = alpha, beta
    prevd, postd, prevp, postp = 0., 0., 0., 0.
    step_a, step_b = alpha / 1000, beta / 1000
    prevd, pval = scistats.kstest(empirical_cdf, scibeta.cdf, args=(a, b))

    # alpha 값 searching

    while True:
        a -= step_a
        postd, pval = scistats.kstest(empirical_cdf, scibeta.cdf, args=(a, b))
        if prevd < postd:
            step_a *= -0.1
        elif postd / prevd > 0.99999999:
            break
        elif pval >= 0.05:
            return postd, pval, a, b
        prevd = postd

    # beta 값 searching

    while True:
        b -= step_b
        postd, pval = scistats.kstest(empirical_cdf, scibeta.cdf, args=(a, b))
        if prevd < postd:
            step_b *= -0.1
        elif postd / prevd > 0.99999999:
            break
        elif pval >= 0.05:
            return postd, pval, a, b
        prevd = postd

    # if pval < 1e-5:
    #     postd, pval = search_betaparam(empirical_cdf, alpha, beta)

    return postd, pval, a, b


def fit_betaparam_search(empirical_cdf, alpha, beta):
    # 앞에서부터 p-value가 0.05미만으로 떨어질 때마다 search로 맞춰준다.
    a, b = alpha, beta
    len_ecdf = len(empirical_cdf)
    stub_d, stub_pval = 0., 0.
    for i in range(1, len_ecdf + 1):
        ecdf_stub = empirical_cdf[:i]
        stub_d, stub_pval = scistats.kstest(ecdf_stub, scibeta.cdf, args=(a, b))
        if stub_pval > 0.05:
            continue
        # stub_d, stub_pval, a, b = search_betaparam(ecdf_stub, a, b)
    return stub_d, stub_pval, a, b