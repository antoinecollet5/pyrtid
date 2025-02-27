import pytest
from pyrtid.inverse.regularization import get_l_curvature


@pytest.mark.parametrize(
    "reg_weights,loss_ls_list,loss_reg_list,is_logspace,"
    "nb_interp_points, expected_interp_l_argmin",
    (
        (
            [
                6.30957344e-02,
                1.06800043e-01,
                1.80776868e-01,
                3.05994969e-01,
                5.17947468e-01,
                8.76712387e-01,
                1.48398179e00,
                2.51188643e00,
                4.25178630e00,
                7.19685673e00,
                1.21818791e01,
                2.06198601e01,
                3.49025488e01,
                5.90783791e01,
                1.00000000e02,
            ],
            [
                3.861131978018497,
                10.63749473306963,
                28.084959701062452,
                70.69602212274012,
                148.7271550741828,
                224.22914633404685,
                268.9491026901526,
                309.97768447738747,
                379.84643748298276,
                552.8606472848202,
                719.6416084899397,
                819.3624621801692,
                908.6353256480135,
                979.3524217244567,
                1031.6536479179792,
            ],
            [
                990.1581263101825,
                910.1187746213185,
                787.9412429461486,
                610.427687199538,
                414.37561084401216,
                294.8650133345532,
                250.68920065400414,
                225.00025261766436,
                196.38186303454313,
                152.70014284103928,
                124.27497506017458,
                110.84896064526771,
                101.29164537597256,
                96.22722897059654,
                92.12759358482776,
            ],
            False,
            100,
            18,
        ),
        (
            [
                1.00000000e00,
                3.16227766e00,
                1.00000000e01,
                3.16227766e01,
                1.00000000e02,
                3.16227766e02,
                1.00000000e03,
                3.16227766e03,
                1.00000000e04,
                3.16227766e04,
                1.00000000e05,
                3.16227766e05,
                1.00000000e06,
                3.16227766e06,
                1.00000000e07,
            ],
            [
                387.50947457257075,
                385.5285277681472,
                386.43816453250327,
                387.527924205149,
                391.0742843245452,
                399.02403165621206,
                421.71006142066096,
                536.1696780756968,
                968.134056565941,
                1922.5807414560654,
                3491.8333379841524,
                5772.763713525066,
                7957.459092686562,
                28152.585268865576,
                91454.4237648074,
            ],
            [
                2.862505763643463,
                1.5536585882247322,
                0.853102247448071,
                0.4541226647106095,
                0.25349817760878024,
                0.18397043963104903,
                0.15086059805537425,
                0.10928008740540246,
                0.07648287655048368,
                0.03996705738919637,
                0.03694392490966868,
                0.031023992018289207,
                0.028403701941107782,
                0.018371651746488117,
                0.009080712162008805,
            ],
            True,
            1000,
            63,
        ),
    ),
)
def test_get_l_curvature(
    reg_weights,
    loss_ls_list,
    loss_reg_list,
    is_logspace,
    nb_interp_points,
    expected_interp_l_argmin,
) -> None:
    (
        interp_reg_weights,
        interp_loss_ls,
        interp_loss_reg,
        lcurve_curvature,
        interp_l_argmin,
    ) = get_l_curvature(
        reg_weights,
        loss_ls_list,
        loss_reg_list,
        is_logspace=is_logspace,
        nb_interp_points=nb_interp_points,
    )
    assert interp_l_argmin == expected_interp_l_argmin
