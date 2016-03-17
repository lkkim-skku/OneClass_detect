"""
model의 distribution과 관련된 클래스와 함수를 정의합니다.
beta, fld등이 있습니다. FLD는 아직...없습니다.
"""
from scipy.stats import beta as scipybeta
from scipy.stats import kstest
from model.dataprocess import *


opt_estimator = 'estimator'
param_mm = 'moment matching'
param_sf = 'sample fit'
param_ct = 'custom'
param_ms = 'moment searching'


class BetaEstimator:
    """
    Beta Model을 modeling합니다.
    option은 다음과 같습니다.

    - estimator: beta parameter를 estimate할 수 있는 알고리즘을 고릅니다.\n
    내장된 estimator는 문자열로 선택할 수 있으며, 개별적으로 만든 estimator는 다음의 형식을 따르면 됩니다.::
    def myestimator(*args): return alpha, beta

    개별적으로 만든 estimator는 다음의 예시로 적용할 수 있습니다.::
    BetaEstimator(estimator=myestimator)

    - alpha: beta parameter 중 alpha입니다.

    - beta: beta parameter 중 beta입니다.

    Oshiete! 먄약 alpha와 beta 모두 입력한다면 어떻게 해야할까요?
    """
    def __init__(self, estimator=param_mm):
        if hasattr(estimator, '__call__'):
            self._estimator = param_ct
            self.__estimator__ = estimator
            self._istimator = False
        else:
            self._estimator = estimator
            self.__estimator__ = BetaEstimator.__estimator_select__(self._estimator)
            self._istimator = True
        self._alpha, self._beta = 1, 1
        self._data = []
        self._empirical = []
        self._betaobj = scipybeta  # TODO 나중에 combinedbeta를 만들 계획입니다.

    @property
    def data(self):
        return self._data

    @property
    def empirical(self):
        return self._empirical

    def parameter(self):
        return self._alpha, self._beta

    def __betafit__(self, alpha, beta):
        """
        beta parameter인 alpha와 beta를 보다 rv frequenct에 최적화시킵니다.

        :param alpha:
        :param beta:
        :return:
        """
        """
        searching으로 beta parameter를 찾습니다.
        moment matching 기반입니다.

        :param BaseData data: sample data
        :return:
        """
        max_try_cnt = 100
        asign, bsign = 1, 1

        step_a, step_b = alpha / 2, alpha / 2
        a, b = alpha, beta

        p, d, try_cnt = 0, 0, 0
        while p < 0.05:
            a += asign * step_a
            p, d = kstest(self._empirical, self._betaobj.cdf, (a, b), N=len(self._empirical))

            try_cnt += 1
            if try_cnt == max_try_cnt:
                break

        p, d, try_cnt = 0, 0, 0
        while p < 0.05:
            b += asign * step_b
            p, d = kstest(self._empirical, self._betaobj.cdf, (a, b), N=len(self._empirical))

            try_cnt += 1
            if try_cnt == max_try_cnt:
                break

        return a, b

    def fit(self, data: BaseData):
        """
        beta parameter를 fit합니다.
        fitting은 다음의 과정을 거칩니다.

        #. rv의 histogram을 구합니다.
        #. histogram of rv의 CDF를 구합니다.
        #. histogram CDF of rv의 각 bin이 나타내는 Frequency(histogram of rv의 y축 값)를 구합니다.
        이것은 rv frequency가 됩니다.
          - 논문 상에서는 histogram of rv가 아니라 CDF of rv입니다. 하지만 CDF of rv는
          KS-test에 부적합할 수 있습니다. 너무 많기 때문이죠. 따라서 이를 피하기 위해 우리는 100개로 한정합니다.
          100개로 한정하려면 어떻게 하는 게 좋을까요? 간단합니다. histogram의 bins를 100개로 고정하면 됩니다.
        #. beta의 parameter를 계산합니다. 이는 initial parameter입니다.
        #. initial parameter를 beta function이 rv frequency를 잘 표현할 수 있도록 fitting합니다.
        #. KS-test 수행 결과 중 하나인 p-value가 0.05 이상이면 accept, 이하면 다시 fitting합니다.

        :param BaseData data: fitting할 대상의 data입니다.
        :return: :class:`BetaEstimator`
        """
        max_try_cnt = 1000

        self._empirical = histo_cudif(data.data, 100)
        a, b = self.__estimator__(data)
        a, b = self.__betafit__(a, b)

        try_count = 0
        p, d = kstest(self._empirical, self._betaobj.cdf, (a, b), N=len(self._empirical))

        while p < 0.05:
            a, b = self.__estimator__(data)
            a, b = self.__betafit__(a, b)
            p, d = kstest(self._empirical, self._betaobj.cdf, (a, b), N=len(self._empirical))

            try_count += 1
            if try_count == max_try_cnt:
                break
        self._alpha, self._beta = a, b
        return self

    def predict(self, ranvar):
        """
        kernelized random variable을 입력 받으면, betaCDF에 대입했을 때의 출력을 반환합니다.
        :param ranvar:
        :return:
        """
        betaoutput = self._betaobj.cdf(ranvar, self._alpha, self._beta)
        return betaoutput

    @staticmethod
    def __estimator_select__(estimator_name=param_mm):
        """
        내장된 estimator를 선택합니다. 다음은 estimator 목록입니다.

        - :func:`be_momentmatch` : moment matching

        - :func:`besamplefitting` : Fitting Beta Distribution Based on Sample Data에서 제안한
        beta fitting 기법입니다. 최초 parameter는 :func:`be_momentmatch` 으로 예측합니다.

        :param str estimator_name: 내장된 estimator를 나타내는 이름
        :return: function address
        """
        if estimator_name in param_sf:
            return be_samplefitting
        else:
            return be_momentmatch
    pass


def be_momentmatch(data: BaseData):
    """
    approximate beta parameters with mean and variance

    :param BaseData data: sample data
    :return:
    """
    mean, var = data.mean, data.var
    lower, upper = 0., 1.  # feature scaling한 뒤니까 당연히 0과 1이죠.
    ml = mean - lower
    um = upper - mean
    a = (ml / (upper - lower)) * (((ml * um) / var) - 1)
    b = a * (um / ml)

    return a, b


def be_samplefitting(data: BaseData):
    """
    Fitting Beta Distributions Based on Sample Data

    :param BaseData data: sample data
    :return:
    """
    alpha, beta = be_momentmatch(data)

    # TODO 여기서부터 본격적인 samplefitting입니다.

    return alpha, beta
