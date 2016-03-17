"""
실험을 위해 임시로 만든 model들을 정의합니다.
"""

from model.dataprocess import normalization as normaller
from model.dataprocess import clustering


class FakeCentroid(clustering.AbstractCluster):
    """
    data를 그대로 list로 감싸서 predict로 만들어버립니다.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__normalizer__ = normaller.FeatureScaler

    def fit(self, data):
        super().fit(data)

    def predict(self):
        """
        centroid의 기본은

        1. 나눈다(clustering)

        2. feature scaling

        입니다.

        :return:
        """
        normed = self.__normalizer__().fit(self._data)
        clustee = clustering.BaseData()
        clustee.fit(normed)
        return [clustee]
        pass
