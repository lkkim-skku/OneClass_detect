__author__ = 'lk'

# from cppy import cpon, cspace
# from lkpy import lkexporter as xprt
import simagent, iomanager
import leesuk


if __name__ == '__main__':
    # cm = NIPSimulator.ClfSim()  # default : 2
    simulation = simagent.SimAgent(fold=10)  # default : 2
    # cm.addclf(NIPSimulator.clffactory('svm'))
    # cm.addclf(NIPSimulator.clffactory('knn'))
    cpon = simagent.clffactory('cpon', cluster='lk', beta='scipy', bse='mm', threadable=False)
    simulation.addsim(cpon)

    data, target = iomanager.load_data()
    # data, target = iomanager.import_data()
    # _data, _target = [], []
    # for d, t in zip(data, target):
    #     if(t == '05' or t == '42'):
    #         _data.append(d)
    #         _target.append(t)
    # data, target = _data, _target

    simulation.unknown = True
    ## simulation.fit(data, target)
    # cm.fit_unknown(data, target, ratio=0.1)

    # 여기서 learning data와 examining data를 관리해줘야 합니다.
    lm = simagent.LearningManager(2)
    # lm.field(data, target)
    lc, lt, ec, et = leesuk.folding_160311(data, target)
    lm.uploadlearn(lc, lt)
    lm.uploadexam(ec, et)
    simulation.folder = lm

    lationresult = simulation.simulate()

    iomanager.measurement(lationresult)
    for cponpst in lationresult:
        if 'cpon' in cponpst.simulorname:
            iomanager.p_value(cponpst)
            # print(cponpst.simulor.pred_pval)
            # for i, ppv_fold in enumerate(cponpst.simulor.pred_pval):
            #     print(("fold %02d" % i) + "p-value result")
            #     ppvstr = ""
            #     for ppv_cls in ppv_fold:
            #         ppvstr += ppv_cls + "\t" + repr(ppv_fold[ppv_cls]) + "\t"
            #     print(ppvstr)

    print("End py")
