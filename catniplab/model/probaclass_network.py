from controller import *
import model
from model.dataprocess import *

opt_ddof = 'ddof'
opt_kernel = 'kernel'
opt_weight = 'weight'


class PCMember:
    """
    :clss:`ProbaClass`의 member class입니다. ML의 view에서 설명하자면 kernel입니다.
    """
    def __init__(self, **kwargs):
        """

        :param kwargs:
        - ddof: data가 sample이면 1, mother popular면 0을 입력합니다.
        - kernel: kernel의 종류를 선택합니다. kernel은 gaussian과 epanechnikov가 있습니다.

        :return:
        """
        self._opts = {
            opt_kernel: kwargs[opt_kernel] if opt_kernel in kwargs else 'gaussian',
            opt_ddof: kwargs[opt_ddof] if opt_ddof in kwargs else 1,
            opt_weight: kwargs[opt_weight] if opt_weight in kwargs else 1
        }
        self._kernel = GeneralKernel(self._opts[opt_kernel], weight=self._opts[opt_weight])
        self._data = BaseData()
        self._beta = model.BetaEstimator()  # moment matching
        # self._beta = model.BetaEstimator('sample_fitting')  # fitting using sample data
        self._data_kernelized = BaseData()  # ABCKernel()

    def fit(self, clustee: BaseData):
        """
        하나의 cluster/kernel에 대한 learning입니다.

        :param model.AbstractCluster clustee:
        :return:
        """

        self._data = clustee
        self._kernel.fit(clustee.mean, clustee.std)
        self._data_kernelized.fit([self._kernel.map(v) for v in clustee.data])
        self._beta.fit(self._data_kernelized)

    def predict(self, ranvar):
        """
        ranvar에 대한 kernel의 betaCDF값을 구합니다.
        :param ranvar:
        :return:
        """
        k = self._kernel.map(ranvar)
        b = self._beta.predict(k)
        return b

    def data(self):
        return self._data

    def kernel(self):
        return self._kernel

    def beta(self):
        return self._beta
    pass


class ProbaClass:
    """
    Class Probability를 나타냅니다.
    """
    def __init__(self, tag):
        self._tag = tag
        # self._data = model.AbstractCluster()  # 만약 구조가 바뀐다면 이걸 쓸 지도...
        self._data = model.FakeCentroid()  # centroid가 1개인 centroid.
        # self._data = model.Centroid()  # 논문상의 clustering 모델 TODO Centroid 마저 만들렴
        self._clustees = []
        self._members = []

    @property
    def tag(self):
        return self._tag

    def fit(self, data: list or tuple):
        """
        넌 뭘 해야되니?

        - clustering
        - learning: kernelize, beta fitting

        :param data:
        :return:
        """
        self._data.fit(data)
        self._clustees = self._data.predict()

        for clustee in self._clustees:
            # 이 d는 StatObject입니다.
            member = PCMember(kernel='gaussian', ddof=1)
            member.fit(clustee)  # 이게 learning입니다.
            self._members.append(member)

        pass

    def predict(self, ranvar):
        """
        :param ranvar: test하기 위한 random variable입니다.
        이 RV의 수학적 정의는 pattern입니다...라뇨 뭔소리죠?
        쨋던, 이 함수는 rv를 입력받으면, 현재 class의 kernel output을 beta function에 대입하여
        그 결과인 beta output을 확보하고 , 그 결과 중에서 가낭 높은 확률을 리턴합니다.

        :return: probability of target.
        """
        membres = []
        for member in self._members:
            membres.append(member.predict(ranvar))

        p = max(membres)  # TODO 몇번째 member인지도 알아야 하나?

        return p
    pass


class CPON:
    """
    Class Probability Output Network입니다.\n
    :class:`ProbaClass` network를 생성한 후 학습(:func:`CPON.fit`)합니다.
    또한 fit된 이후에는 예측(:func:`cpon.predict`)할 수 있습니다.

    ※CPON output과 class output의 차이점 \n
    class output은 :class:`ProbaClass` 의 예측값, 즉 :func:`ProbaClass.predict` 입니다. betaCDF로 구한 확률값이죠.\n
    CPON output은 :class:`ProbaClass` network에 속한 모든 :class:`ProbaClass` 의 예측값을 모아서
    별도로 확률을 계산합니다. 이 때 가장 확률값이 높은 :class:`ProbaClass` 로 예측합니다.
    """
    def __init__(self, **kwargs):
        self._pcnetwork = {}
        self._classoutputs_network = []
        self._ispredicted = False  # output network가 생성되어 있는지를 확인합니다.
        self._isfitted = False
        self._opts = kwargs

    def fit(self, data, target):
        kv = classionary(data, target)
        try:
            for k in kv:
                self._pcnetwork[k] = ProbaClass(k)
                self._pcnetwork[k].fit(kv[k])
        except KeyError:
            self._isfitted = False
            raise(KeyError("unexpected key"))
        finally:
            self._ispredicted = False

        self._isfitted = True
        pass

    # TODO 꼭 수정해야 합니다. 이거 나을텍 전용입니다.
    def predict(self, data):
        """
        예측(predict/examinate/test) 합니다.

        :param data: array-like
        :return: array-like. predict result. 예측한 class tag에 대한 list입니다.
        """
        if not self._isfitted:
            raise(AttributeError("fit first."))
        # TODO 꼭 수정해야 합니다. 이거 나을텍 전용입니다.

        self._classoutputs_network = []
        self.predict_classes(data)
        _pred = []

        for cpnet in self._classoutputs_network:
            _pred.append(mymax(cpnet, key=lambda x: cpnet[x], default='unknown'))

        # TODO 꼭 수정해야 합니다. 이거 나을텍 전용입니다.
        return _pred

    def predict_classes(self, data):
        """
        class output들을 구합니다.
        이 함수는 모든 test value에 대한 모든 class output들을 계산합니다.
        만약 각 value에 대한 predict가 가장 큰 ProbaClass의 tag를 알고 싶다면, :func:`mymax` 를 활용하길 추천합니다.
        :param data: array-like
        :return:
        """
        # TODO 지금 이건 나을텍용입니다.
        for d in data:
            self._classoutputs_network.append(self.__naultech__(d))
            # self._classoutputs_network.append(self.__output__(d))

        self._ispredicted = True

    def __naultech__(self, pattern):
        """
        한 개의 패턴에 대한 predict
        :param pattern:
        :return:
        """
        class_outputs = {}

        for key, value in zip(self._pcnetwork, pattern):
            pc = self._pcnetwork[key]
            class_outputs[key] = pc.predict(value)

        return class_outputs

    def __output__(self, value):
        """
        class output을 계산합니다.

        :param value: cpon에 물어볼 데이터
        :return dict: dictionary of outputs. 각 :class:`Probaclass`의 출력.
        """
        class_outputs = {}

        for key in self._pcnetwork:
            pc = self._pcnetwork[key]
            class_outputs[key] = pc.predict(value)

        return class_outputs
    pass
