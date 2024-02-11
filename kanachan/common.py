import torch
from torch import Tensor


def symlog(x: Tensor) -> Tensor:
    return x.sgn() * (x.abs() + 1.0).log()


def symexp(x: Tensor) -> Tensor:
    return x.sgn() * (x.abs().exp() - 1.0)


def piecewise_linear_encoding(
    value: float | int | Tensor,
    minimum: float,
    maximum: float,
    dimension: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if isinstance(value, float):
        pass
    elif isinstance(value, int):
        value = float(value)
    elif isinstance(value, Tensor):
        if value.dim() not in (0, 1):
            raise ValueError(value)
        if value.dim() == 1 and value.size(0) != 1:
            raise ValueError(value)
        value = float(value.item())
    else:
        raise ValueError(value)
    if minimum > maximum:
        errmsg = f"{minimum} > {maximum}"
        raise ValueError(errmsg)
    if dimension <= 0:
        raise ValueError(dimension)

    interval = (maximum - minimum) / dimension
    intervals = torch.arange(
        1,
        dimension + 1,
        requires_grad=False,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    intervals *= interval
    intervals += minimum
    result = torch.full_like(intervals, value)
    result = torch.where(
        result >= intervals, torch.ones_like(result), torch.zeros_like(result)
    )

    value = min(max(value, minimum), maximum)
    bin_integral = int((value - minimum) / interval)
    assert bin_integral >= 0
    assert bin_integral <= dimension
    bin_fractional = (value - minimum) / interval - bin_integral
    if bin_fractional > 0.0:
        assert bin_integral < dimension
        assert result[bin_integral].item() == 0.0
        result[bin_integral] = bin_fractional

    return result.to(device=device, dtype=dtype)


_REWARD_BASE = [
    [  # 銅の間
        [  # 東風戦
            [  # 初心1
                25,
                10,
                -5,
                -15,
            ],
            [  # 初心2
                25,
                10,
                -5,
                -15,
            ],
            [  # 初心3
                25,
                10,
                -5,
                -15,
            ],
            [  # 雀士1
                25,
                10,
                -5,
                -25,
            ],
            [  # 雀士2
                25,
                10,
                -5,
                -35,
            ],
            [  # 雀士3
                25,
                10,
                -5,
                -45,
            ],
            None,  # 雀傑1
            None,  # 雀傑2
            None,  # 雀傑3
            None,  # 雀豪1
            None,  # 雀豪2
            None,  # 雀豪3
            None,  # 雀聖1
            None,  # 雀聖2
            None,  # 雀聖3
            None,  # 魂天
        ],
        [  # 半荘戦
            [  # 初心1
                35,
                15,
                -5,
                -15,
            ],
            [  # 初心2
                35,
                15,
                -5,
                -15,
            ],
            [  # 初心3
                35,
                15,
                -5,
                -15,
            ],
            [  # 雀士1
                35,
                15,
                -5,
                -35,
            ],
            [  # 雀士2
                35,
                15,
                -5,
                -55,
            ],
            [  # 雀士3
                35,
                15,
                -5,
                -75,
            ],
            None,  # 雀傑1
            None,  # 雀傑2
            None,  # 雀傑3
            None,  # 雀豪1
            None,  # 雀豪2
            None,  # 雀豪3
            None,  # 雀聖1
            None,  # 雀聖2
            None,  # 雀聖3
            None,  # 魂天
        ],
    ],
    [  # 銀の間
        [  # 東風戦
            None,  # 初心1
            None,  # 初心2
            None,  # 初心3
            [  # 雀士1
                35,
                15,
                -5,
                -25,
            ],
            [  # 雀士2
                35,
                15,
                -5,
                -35,
            ],
            [  # 雀士3
                35,
                15,
                -5,
                -45,
            ],
            [  # 雀傑1
                35,
                15,
                -5,
                -55,
            ],
            [  # 雀傑2
                35,
                15,
                -5,
                -65,
            ],
            [  # 雀傑3
                35,
                15,
                -5,
                -75,
            ],
            None,  # 雀豪1
            None,  # 雀豪2
            None,  # 雀豪3
            None,  # 雀聖1
            None,  # 雀聖2
            None,  # 雀聖3
            None,  # 魂天
        ],
        [  # 半荘戦
            None,  # 初心1
            None,  # 初心2
            None,  # 初心3
            [  # 雀士1
                55,
                25,
                -5,
                -35,
            ],
            [  # 雀士2
                55,
                25,
                -5,
                -55,
            ],
            [  # 雀士3
                55,
                25,
                -5,
                -75,
            ],
            [  # 雀傑1
                55,
                25,
                -5,
                -95,
            ],
            [  # 雀傑2
                55,
                25,
                -5,
                -115,
            ],
            [  # 雀傑3
                55,
                25,
                -5,
                -135,
            ],
            None,  # 雀豪1
            None,  # 雀豪2
            None,  # 雀豪3
            None,  # 雀聖1
            None,  # 雀聖2
            None,  # 雀聖3
            None,  # 魂天
        ],
    ],
    [  # 金の間
        [  # 東風戦
            None,  # 初心1
            None,  # 初心2
            None,  # 初心3
            None,  # 雀士1
            None,  # 雀士2
            None,  # 雀士3
            [  # 雀傑1
                55,
                25,
                -5,
                -55,
            ],
            [  # 雀傑2
                55,
                25,
                -5,
                -65,
            ],
            [  # 雀傑3
                55,
                25,
                -5,
                -75,
            ],
            [  # 雀豪1
                55,
                25,
                -5,
                -95,
            ],
            [  # 雀豪2
                55,
                25,
                -5,
                -105,
            ],
            [  # 雀豪3
                55,
                25,
                -5,
                -115,
            ],
            None,  # 雀聖1
            None,  # 雀聖2
            None,  # 雀聖3
            None,  # 魂天
        ],
        [  # 半荘戦
            None,  # 初心1
            None,  # 初心2
            None,  # 初心3
            None,  # 雀士1
            None,  # 雀士2
            None,  # 雀士3
            [  # 雀傑1
                95,
                45,
                -5,
                -95,
            ],
            [  # 雀傑2
                95,
                45,
                -5,
                -115,
            ],
            [  # 雀傑3
                95,
                45,
                -5,
                -135,
            ],
            [  # 雀豪1
                95,
                45,
                -5,
                -180,
            ],
            [  # 雀豪2
                95,
                45,
                -5,
                -195,
            ],
            [  # 雀豪3
                95,
                45,
                -5,
                -210,
            ],
            None,  # 雀聖1
            None,  # 雀聖2
            None,  # 雀聖3
            None,  # 魂天
        ],
    ],
    [  # 玉の間
        [  # 東風戦
            None,  # 初心1
            None,  # 初心2
            None,  # 初心3
            None,  # 雀士1
            None,  # 雀士2
            None,  # 雀士3
            None,  # 雀傑1
            None,  # 雀傑2
            None,  # 雀傑3
            [  # 雀豪1
                70,
                35,
                -5,
                -95,
            ],
            [  # 雀豪2
                70,
                35,
                -5,
                -105,
            ],
            [  # 雀豪3
                70,
                35,
                -5,
                -115,
            ],
            [  # 雀聖1
                70,
                35,
                -5,
                -125,
            ],
            [  # 雀聖2
                70,
                35,
                -5,
                -135,
            ],
            [  # 雀聖3
                70,
                35,
                -5,
                -145,
            ],
            None,  # 魂天
        ],
        [  # 半荘戦
            None,  # 初心1
            None,  # 初心2
            None,  # 初心3
            None,  # 雀士1
            None,  # 雀士2
            None,  # 雀士3
            None,  # 雀傑1
            None,  # 雀傑2
            None,  # 雀傑3
            [  # 雀豪1
                125,
                60,
                -5,
                -180,
            ],
            [  # 雀豪2
                125,
                60,
                -5,
                -195,
            ],
            [  # 雀豪3
                125,
                60,
                -5,
                -210,
            ],
            [  # 雀聖1
                125,
                60,
                -5,
                -225,
            ],
            [  # 雀聖2
                125,
                60,
                -5,
                -240,
            ],
            [  # 雀聖3
                125,
                60,
                -5,
                -255,
            ],
            None,  # 魂天
        ],
    ],
    [  # 王座の間
        [  # 東風戦
            None,  # 初心1
            None,  # 初心2
            None,  # 初心3
            None,  # 雀士1
            None,  # 雀士2
            None,  # 雀士3
            None,  # 雀傑1
            None,  # 雀傑2
            None,  # 雀傑3
            None,  # 雀豪1
            None,  # 雀豪2
            None,  # 雀豪3
            [  # 雀聖1
                75,
                35,
                -5,
                -125,
            ],
            [  # 雀聖2
                75,
                35,
                -5,
                -135,
            ],
            [  # 雀聖3
                75,
                35,
                -5,
                -145,
            ],
            [  # 魂天
                0.3,
                0.1,
                -0.1,
                -0.3,
            ],
        ],
        [  # 半荘戦
            None,  # 初心1
            None,  # 初心2
            None,  # 初心3
            None,  # 雀士1
            None,  # 雀士2
            None,  # 雀士3
            None,  # 雀傑1
            None,  # 雀傑2
            None,  # 雀傑3
            None,  # 雀豪1
            None,  # 雀豪2
            None,  # 雀豪3
            [  # 雀聖1
                135,
                65,
                -5,
                -225,
            ],
            [  # 雀聖2
                135,
                65,
                -5,
                -240,
            ],
            [  # 雀聖3
                135,
                65,
                -5,
                -255,
            ],
            [  # 魂天
                0.5,
                0.2,
                -0.2,
                -0.5,
            ],
        ],
    ],
]


def get_grading_point(
    room: int,
    game_style: int,
    grade: int,
    ranking: int,
    score: int,
    celestial_scale: int,
    all_celestial: bool,
) -> float:
    if room < 0 or 4 < room:
        raise ValueError(room)
    if game_style not in (0, 1):
        raise ValueError(game_style)
    if grade < 0 or 15 < grade:
        raise ValueError(grade)
    if ranking < 0 or 3 < ranking:
        raise ValueError(ranking)
    if celestial_scale <= 0:
        raise ValueError(celestial_scale)

    grading_point = _REWARD_BASE[room][game_style][grade][ranking]
    if grading_point is None:
        errmsg = "An invalid combination of the room and the grade."
        raise RuntimeError(errmsg)
    if grade == 15:
        grading_point *= celestial_scale
    else:
        grading_point += (score - 25000) // 1000
    if all_celestial:
        grading_point *= 2

    return float(grading_point)
