"""

Y M - 나 을 텍 전 용 메 인

"""

__author__ = 'lkkim'


from controller.projio import localio
from controller import simagent

if __name__ == '__main__':
    """
    자 보자...
    일단 우리가 사용하는 데이터는 classifier output이야. 그러니까
    1. Y에 대한 normalize을 해야한다.
    2. beta approximation을 한다.
    3. kstest로 결과 확인.
    """

    # Searching을 통해서 최적의 beta function을 찾아라.
    # a = 1/100을 시작으로 \alpha값을 먼저 최적화하고
    # b = 1/100을 시작으로 \beta값을 먼저 최적화한다.
    # \alpha += a부터 시작하고, p값이 다시 감소한다면 a *= 1/10을 하여 다시 최적화한다.
    # \beta도 alpha와 같은 방법으로 찾는다.

    agent = simagent.SimAgent(fold=1)
    cpon = simagent.clffactory('cpon')
    agent.addsim(cpon)

    # 여기서 learning data와 examining data를 관리해줘야 합니다.
    lm = simagent.LearningManager(2)

    learningdata, learningtarget = localio.naultech_learn()
    testingdata, testingtarget = localio.naultech_test()

    lm.uploadlearn([learningdata], [learningtarget])
    lm.uploadexam([testingdata], [testingtarget])

    agent.folder = lm

    predict_result = agent.simulate()
    print("End py")
