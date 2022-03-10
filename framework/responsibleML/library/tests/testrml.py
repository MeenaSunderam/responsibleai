from aigovernance import responsibleML as myModel

def test_rai_functions():
    myModel.set_explainability(True)
    myModel.set_emissions(0.6)
    myModel.set_bias(0.2)
    myModel.set_diff_privacy(True)

    index = 0.0
    index  = myModel.rai_index()
    print(index)
    assert(index)