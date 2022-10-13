#!/usr/bin/env python3

from typing import (List,)
from mahjong.meld import Meld
from mahjong.hand_calculating.hand_config import (OptionalRules, HandConfig,)
from mahjong.hand_calculating.hand import HandCalculator as Impl


_FULU2MELD = {
    148: Meld('kan', [  0,   1,   2,   3], False),
    149: Meld('kan', [  4,   5,   6,   7], False),
    150: Meld('kan', [  8,   9,  10,  11], False),
    151: Meld('kan', [ 12,  13,  14,  15], False),
    152: Meld('kan', [ 16,  17,  18,  19], False),
    153: Meld('kan', [ 20,  21,  22,  23], False),
    154: Meld('kan', [ 24,  25,  26,  27], False),
    155: Meld('kan', [ 28,  29,  30,  31], False),
    156: Meld('kan', [ 32,  33,  34,  35], False),
    157: Meld('kan', [ 36,  37,  38,  39], False),
    158: Meld('kan', [ 40,  41,  42,  43], False),
    159: Meld('kan', [ 44,  45,  46,  47], False),
    160: Meld('kan', [ 48,  49,  50,  51], False),
    161: Meld('kan', [ 52,  53,  54,  55], False),
    162: Meld('kan', [ 56,  57,  58,  59], False),
    163: Meld('kan', [ 60,  61,  62,  63], False),
    164: Meld('kan', [ 64,  65,  66,  67], False),
    165: Meld('kan', [ 68,  69,  70,  71], False),
    166: Meld('kan', [ 72,  73,  74,  75], False),
    167: Meld('kan', [ 76,  77,  78,  79], False),
    168: Meld('kan', [ 80,  81,  82,  83], False),
    169: Meld('kan', [ 84,  85,  86,  87], False),
    170: Meld('kan', [ 88,  89,  90,  91], False),
    171: Meld('kan', [ 92,  93,  94,  95], False),
    172: Meld('kan', [ 96,  97,  98,  99], False),
    173: Meld('kan', [100, 101, 102, 103], False),
    174: Meld('kan', [104, 105, 106, 107], False),
    175: Meld('kan', [108, 109, 110, 111], False),
    176: Meld('kan', [112, 113, 114, 115], False),
    177: Meld('kan', [116, 117, 118, 119], False),
    178: Meld('kan', [120, 121, 122, 123], False),
    179: Meld('kan', [124, 125, 126, 127], False),
    180: Meld('kan', [128, 129, 130, 131], False),
    181: Meld('kan', [132, 133, 134, 135], False),

    182: Meld('kan', [ 16,  17,  18,  19]),
    183: Meld('kan', [  0,   1,   2,   3]),
    184: Meld('kan', [  4,   5,   6,   7]),
    185: Meld('kan', [  8,   9,  10,  11]),
    186: Meld('kan', [ 12,  13,  14,  15]),
    187: Meld('kan', [ 16,  17,  18,  19]),
    188: Meld('kan', [ 20,  21,  22,  23]),
    189: Meld('kan', [ 24,  25,  26,  27]),
    190: Meld('kan', [ 28,  29,  30,  31]),
    191: Meld('kan', [ 32,  33,  34,  35]),
    192: Meld('kan', [ 52,  53,  54,  55]),
    193: Meld('kan', [ 36,  37,  38,  39]),
    194: Meld('kan', [ 40,  41,  42,  43]),
    195: Meld('kan', [ 44,  45,  46,  47]),
    196: Meld('kan', [ 48,  49,  50,  51]),
    197: Meld('kan', [ 52,  53,  54,  55]),
    198: Meld('kan', [ 56,  57,  58,  59]),
    199: Meld('kan', [ 60,  61,  62,  63]),
    200: Meld('kan', [ 64,  65,  66,  67]),
    201: Meld('kan', [ 68,  69,  70,  71]),
    202: Meld('kan', [ 88,  89,  90,  91]),
    203: Meld('kan', [ 72,  73,  74,  75]),
    204: Meld('kan', [ 76,  77,  78,  79]),
    205: Meld('kan', [ 80,  81,  82,  83]),
    206: Meld('kan', [ 84,  85,  86,  87]),
    207: Meld('kan', [ 88,  89,  90,  91]),
    208: Meld('kan', [ 92,  93,  94,  95]),
    209: Meld('kan', [ 96,  97,  98,  99]),
    210: Meld('kan', [100, 101, 102, 103]),
    211: Meld('kan', [104, 105, 106, 107]),
    212: Meld('kan', [108, 109, 110, 111]),
    213: Meld('kan', [112, 113, 114, 115]),
    214: Meld('kan', [116, 117, 118, 119]),
    215: Meld('kan', [120, 121, 122, 123]),
    216: Meld('kan', [124, 125, 126, 127]),
    217: Meld('kan', [128, 129, 130, 131]),
    218: Meld('kan', [132, 133, 134, 135]),

    222: Meld('chi', [  0,   4,   8]),
    223: Meld('chi', [  0,   4,   8]),
    224: Meld('chi', [  4,   8,  12]),
    225: Meld('chi', [  0,   4,   8]),
    226: Meld('chi', [  4,   8,  12]),
    227: Meld('chi', [  8,  12,  17]),
    228: Meld('chi', [  8,  12,  16]),
    229: Meld('chi', [  4,   8,  12]),
    230: Meld('chi', [  8,  12,  17]),
    231: Meld('chi', [  8,  12,  16]),
    232: Meld('chi', [ 12,  17,  20]),
    233: Meld('chi', [ 12,  16,  20]),
    234: Meld('chi', [  8,  12,  17]),
    235: Meld('chi', [  8,  12,  16]),
    236: Meld('chi', [ 12,  17,  20]),
    237: Meld('chi', [ 12,  16,  20]),
    238: Meld('chi', [ 17,  20,  24]),
    239: Meld('chi', [ 16,  20,  24]),
    240: Meld('chi', [ 12,  17,  20]),
    241: Meld('chi', [ 12,  16,  20]),
    242: Meld('chi', [ 17,  20,  24]),
    243: Meld('chi', [ 16,  20,  24]),
    244: Meld('chi', [ 20,  24,  28]),
    245: Meld('chi', [ 17,  20,  24]),
    246: Meld('chi', [ 16,  20,  24]),
    247: Meld('chi', [ 20,  24,  28]),
    248: Meld('chi', [ 24,  28,  32]),
    249: Meld('chi', [ 20,  24,  28]),
    250: Meld('chi', [ 24,  28,  32]),
    251: Meld('chi', [ 24,  28,  32]),
    252: Meld('chi', [ 36,  40,  44]),
    253: Meld('chi', [ 36,  40,  44]),
    254: Meld('chi', [ 40,  44,  48]),
    255: Meld('chi', [ 36,  40,  44]),
    256: Meld('chi', [ 40,  44,  48]),
    257: Meld('chi', [ 44,  48,  53]),
    258: Meld('chi', [ 44,  48,  52]),
    259: Meld('chi', [ 40,  44,  48]),
    260: Meld('chi', [ 44,  48,  53]),
    261: Meld('chi', [ 44,  48,  52]),
    262: Meld('chi', [ 48,  53,  56]),
    263: Meld('chi', [ 48,  52,  56]),
    264: Meld('chi', [ 44,  48,  53]),
    265: Meld('chi', [ 44,  48,  52]),
    266: Meld('chi', [ 48,  53,  56]),
    267: Meld('chi', [ 48,  52,  56]),
    268: Meld('chi', [ 53,  56,  60]),
    269: Meld('chi', [ 52,  56,  60]),
    270: Meld('chi', [ 48,  53,  56]),
    271: Meld('chi', [ 48,  52,  56]),
    272: Meld('chi', [ 53,  56,  60]),
    273: Meld('chi', [ 52,  56,  60]),
    274: Meld('chi', [ 56,  60,  64]),
    275: Meld('chi', [ 53,  56,  60]),
    276: Meld('chi', [ 52,  56,  60]),
    277: Meld('chi', [ 56,  60,  64]),
    278: Meld('chi', [ 60,  64,  68]),
    279: Meld('chi', [ 56,  60,  64]),
    280: Meld('chi', [ 60,  64,  68]),
    281: Meld('chi', [ 60,  64,  68]),
    282: Meld('chi', [ 72,  76,  80]),
    283: Meld('chi', [ 72,  76,  80]),
    284: Meld('chi', [ 76,  80,  84]),
    285: Meld('chi', [ 72,  76,  80]),
    286: Meld('chi', [ 76,  80,  84]),
    287: Meld('chi', [ 80,  84,  89]),
    288: Meld('chi', [ 80,  84,  88]),
    289: Meld('chi', [ 76,  80,  84]),
    290: Meld('chi', [ 80,  84,  89]),
    291: Meld('chi', [ 80,  84,  88]),
    292: Meld('chi', [ 84,  89,  92]),
    293: Meld('chi', [ 84,  88,  92]),
    294: Meld('chi', [ 80,  84,  89]),
    295: Meld('chi', [ 80,  84,  88]),
    296: Meld('chi', [ 84,  89,  92]),
    297: Meld('chi', [ 84,  88,  92]),
    298: Meld('chi', [ 89,  92,  96]),
    299: Meld('chi', [ 88,  92,  96]),
    300: Meld('chi', [ 84,  89,  92]),
    301: Meld('chi', [ 84,  88,  92]),
    302: Meld('chi', [ 89,  92,  96]),
    303: Meld('chi', [ 88,  92,  96]),
    304: Meld('chi', [ 92,  96, 100]),
    305: Meld('chi', [ 89,  92,  96]),
    306: Meld('chi', [ 88,  92,  96]),
    307: Meld('chi', [ 92,  96, 100]),
    308: Meld('chi', [ 96, 100, 104]),
    309: Meld('chi', [ 92,  96, 100]),
    310: Meld('chi', [ 96, 100, 104]),
    311: Meld('chi', [ 96, 100, 104]),

    312: Meld('pon', [  0,   1,   2]),
    313: Meld('pon', [  4,   5,   6]),
    314: Meld('pon', [  8,   9,  10]),
    315: Meld('pon', [ 12,  13,  14]),
    316: Meld('pon', [ 17,  18,  19]),
    317: Meld('pon', [ 16,  17,  18]),
    318: Meld('pon', [ 16,  17,  18]),
    319: Meld('pon', [ 20,  21,  22]),
    320: Meld('pon', [ 24,  25,  26]),
    321: Meld('pon', [ 28,  29,  30]),
    322: Meld('pon', [ 32,  33,  34]),
    323: Meld('pon', [ 36,  37,  38]),
    324: Meld('pon', [ 40,  41,  42]),
    325: Meld('pon', [ 44,  45,  46]),
    326: Meld('pon', [ 48,  49,  50]),
    327: Meld('pon', [ 53,  54,  55]),
    328: Meld('pon', [ 52,  53,  54]),
    329: Meld('pon', [ 52,  53,  54]),
    330: Meld('pon', [ 56,  57,  58]),
    331: Meld('pon', [ 60,  61,  62]),
    332: Meld('pon', [ 64,  65,  66]),
    333: Meld('pon', [ 68,  69,  70]),
    334: Meld('pon', [ 72,  73,  74]),
    335: Meld('pon', [ 76,  77,  78]),
    336: Meld('pon', [ 80,  81,  82]),
    337: Meld('pon', [ 84,  85,  86]),
    338: Meld('pon', [ 89,  90,  91]),
    339: Meld('pon', [ 88,  89,  90]),
    340: Meld('pon', [ 88,  89,  90]),
    341: Meld('pon', [ 92,  93,  94]),
    342: Meld('pon', [ 96,  97,  98]),
    343: Meld('pon', [100, 101, 102]),
    344: Meld('pon', [104, 105, 106]),
    345: Meld('pon', [108, 109, 110]),
    346: Meld('pon', [112, 113, 114]),
    347: Meld('pon', [116, 117, 118]),
    348: Meld('pon', [120, 121, 122]),
    349: Meld('pon', [124, 125, 126]),
    350: Meld('pon', [128, 129, 130]),
    351: Meld('pon', [132, 133, 134]),

    432: Meld('kan', [ 16,  17,  18,  19]),
    433: Meld('kan', [  0,   1,   2,   3]),
    434: Meld('kan', [  4,   5,   6,   7]),
    435: Meld('kan', [  8,   9,  10,  11]),
    436: Meld('kan', [ 12,  13,  14,  15]),
    437: Meld('kan', [ 16,  17,  18,  19]),
    438: Meld('kan', [ 20,  21,  22,  23]),
    439: Meld('kan', [ 24,  25,  26,  27]),
    440: Meld('kan', [ 28,  29,  30,  31]),
    441: Meld('kan', [ 32,  33,  34,  35]),
    442: Meld('kan', [ 52,  53,  54,  55]),
    443: Meld('kan', [ 36,  37,  38,  39]),
    444: Meld('kan', [ 40,  41,  42,  43]),
    445: Meld('kan', [ 44,  45,  46,  47]),
    446: Meld('kan', [ 48,  49,  50,  51]),
    447: Meld('kan', [ 52,  53,  54,  55]),
    448: Meld('kan', [ 56,  57,  58,  59]),
    449: Meld('kan', [ 60,  61,  62,  63]),
    450: Meld('kan', [ 64,  65,  66,  67]),
    451: Meld('kan', [ 68,  69,  70,  71]),
    452: Meld('kan', [ 88,  89,  90,  91]),
    453: Meld('kan', [ 72,  73,  74,  75]),
    454: Meld('kan', [ 76,  77,  78,  79]),
    455: Meld('kan', [ 80,  81,  82,  83]),
    456: Meld('kan', [ 84,  85,  86,  87]),
    457: Meld('kan', [ 88,  89,  90,  91]),
    458: Meld('kan', [ 92,  93,  94,  95]),
    459: Meld('kan', [ 96,  97,  98,  99]),
    460: Meld('kan', [100, 101, 102, 103]),
    461: Meld('kan', [104, 105, 106, 107]),
    462: Meld('kan', [108, 109, 110, 111]),
    463: Meld('kan', [112, 113, 114, 115]),
    464: Meld('kan', [116, 117, 118, 119]),
    465: Meld('kan', [120, 121, 122, 123]),
    466: Meld('kan', [124, 125, 126, 127]),
    467: Meld('kan', [128, 129, 130, 131]),
    468: Meld('kan', [132, 133, 134, 135])
}


_TILE_OFFSET_RANGE = [
    ( 16,  17),
    (  0,   4),
    (  4,   8),
    (  8,  12),
    ( 12,  16),
    ( 17,  20),
    ( 20,  24),
    ( 24,  28),
    ( 28,  32),
    ( 32,  36),

    ( 52,  53),
    ( 36,  40),
    ( 40,  44),
    ( 44,  48),
    ( 48,  52),
    ( 53,  56),
    ( 56,  60),
    ( 60,  64),
    ( 64,  68),
    ( 68,  72),

    ( 88,  89),
    ( 72,  76),
    ( 76,  80),
    ( 80,  84),
    ( 84,  88),
    ( 89,  92),
    ( 92,  96),
    ( 96, 100),
    (100, 104),
    (104, 108),

    (108, 112),
    (112, 116),
    (116, 120),
    (120, 124),
    (124, 128),
    (128, 132),
    (132, 136)
]


class HandCalculator:
    def __init__(self) -> None:
        pass

    def has_yihan(
        self, chang: int, player_wind: int, hand: List[int],
        fulu_list: List[int], hupai: int, rong: bool) -> bool:
        tiles = set()
        melds = []

        # `tiles` には副露牌も含めなければならない．
        # ただし，槓は3枚としてカウントする．
        for fulu in fulu_list:
            if 148 <= fulu and fulu <= 181:
                meld = _FULU2MELD[fulu]
                for i in range(3):
                    tiles.add(meld.tiles[i])
            elif 182 <= fulu and fulu <= 218:
                meld = _FULU2MELD[fulu]
                for i in range(3):
                    tiles.add(meld.tiles[i])
            elif 222 <= fulu and fulu <= 311:
                meld = _FULU2MELD[fulu]
                for t in meld.tiles:
                    for i in range(4):
                        if t + i not in tiles:
                            tiles.add(t + i)
                            break
            elif 312 <= fulu and fulu <= 431:
                encode = fulu - 312
                peng = encode % 40
                meld = _FULU2MELD[peng + 312]
                for t in meld.tiles:
                    for i in range(4):
                        if t + i not in tiles:
                            tiles.add(t + i)
                            break
            elif 432 <= fulu and fulu <= 542:
                encode = fulu - 432
                daminggang = encode % 37
                meld = _FULU2MELD[daminggang + 432]
                for i in range(3):
                    tiles.add(meld.tiles[i])
            else:
                raise RuntimeError(fulu)
            melds.append(meld)

        for tile in hand:
            flag = False
            first, last = _TILE_OFFSET_RANGE[tile]
            for t in range(first, last):
                if t not in tiles:
                    tiles.add(t)
                    flag = True
                    break
            if not flag:
                raise RuntimeError('TODO: (A suitable error message)')

        # `tiles` には和了牌も含めなければならない．
        flag = False
        first, last = _TILE_OFFSET_RANGE[hupai]
        for t in range(first, last):
            if t not in tiles:
                tiles.add(t)
                flag = True
                break
        if not flag:
            raise RuntimeError('TODO: (A suitable error message)')

        tiles = list(tiles)
        tiles.sort()

        hupai = _TILE_OFFSET_RANGE[hupai][0]

        options = OptionalRules(has_open_tanyao=True, has_aka_dora=True)
        config = HandConfig(
            is_tsumo=not rong, player_wind=27 + player_wind,
            round_wind=27 + chang, options=options)
        hand_calculator = Impl()

        try:
            response = hand_calculator.estimate_hand_value(
                tiles=tiles, win_tile=hupai, melds=melds, config=config)
            if response.error is not None:
                if response.error == 'There are no yaku in the hand':
                    return False
                raise RuntimeError(response.error)
        except Exception as e:
            import sys
            print(f'tiles = {tiles}', file=sys.stderr)
            for meld in melds:
                print(f'meld = {meld.tiles}', file=sys.stderr)
            print(f'hupai = {hupai}', file=sys.stderr)
            raise e

        return response.han >= 1
