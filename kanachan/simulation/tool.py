#!/usr/bin/env python3

from typing import (Tuple, List,)
import sys
from traceback import print_exc
from mahjong.shanten import Shanten
from mahjong.meld import Meld
from mahjong.hand_calculating.hand_config import (OptionalRules, HandConfig)
from mahjong.hand_calculating.hand import HandCalculator


class Tool(object):
    def __init__(self) -> None:
        self.__shanten_calculator = Shanten()
        self.__options = OptionalRules(has_open_tanyao=True, has_aka_dora=True)
        self.__hand_calculator = HandCalculator()

    def _convert_to_win_tile(self, hupai: int) -> int:
        if hupai == 0:
            return 16
        elif 1 <= hupai and hupai <= 4:
            return (hupai - 1) * 4
        elif hupai == 5:
            return 17
        elif 6 <= hupai and hupai <= 9:
            return (hupai - 1) * 4
        elif hupai == 10:
            return 52
        elif 11 <= hupai and hupai <= 14:
            return (hupai - 2) * 4
        elif hupai == 15:
            return 53
        elif 16 <= hupai and hupai <= 19:
            return (hupai - 2) * 4
        elif hupai == 20:
            return 88
        elif 21 <= hupai and hupai <= 24:
            return (hupai - 3) * 4
        elif hupai == 25:
            return 89
        else:
            assert(26 <= hupai and hupai < 37)
            return (hupai - 3) * 4

    def _decode_tool_config(self, tool_config: int) -> HandConfig:
        is_tsumo         = (tool_config & (1 << 0)) != 0
        is_riichi        = (tool_config & (1 << 1)) != 0
        is_ippatsu       = (tool_config & (1 << 2)) != 0
        is_rinshan       = (tool_config & (1 << 3)) != 0
        is_chankan       = (tool_config & (1 << 4)) != 0
        is_haitei        = (tool_config & (1 << 5)) != 0
        is_houtei        = (tool_config & (1 << 6)) != 0
        is_daburu_riichi = (tool_config & (1 << 7)) != 0
        is_tenhou        = (tool_config & (1 << 8)) != 0
        is_chiihou       = (tool_config & (1 << 9)) != 0
        if (tool_config & (1 << 10)) != 0:
            player_wind = 27
        elif (tool_config & (1 << 11)) != 0:
            player_wind = 28
        elif (tool_config & (1 << 12)) != 0:
            player_wind = 29
        elif (tool_config & (1 << 13)) != 0:
            player_wind = 30
        if (tool_config & (1 << 14)) != 0:
            round_wind = 27
        elif (tool_config & (1 << 15)) != 0:
            round_wind = 28
        elif (tool_config & (1 << 16)) != 0:
            round_wind = 29
        elif (tool_config & (1 << 17)) != 0:
            round_wind = 30

        return HandConfig(
            is_tsumo=is_tsumo, is_riichi=is_riichi, is_ippatsu=is_ippatsu,
            is_rinshan=is_rinshan, is_chankan=is_chankan, is_haitei=is_haitei,
            is_houtei=is_houtei, is_daburu_riichi=is_daburu_riichi,
            is_tenhou=is_tenhou, is_chiihou=is_chiihou, player_wind=player_wind,
            round_wind=round_wind, options=self.__options)

    def calculate_xiangting(self, hand: List[int]) -> int:
        try:
            assert isinstance(hand, list)
            assert sum(hand) in (1, 2, 4, 5, 7, 8, 10, 11, 13, 14)
            return self.__shanten_calculator.calculate_shanten(hand)
        except Exception:
            print_exc(file=sys.stderr)
            raise

    _Meld = Tuple[str, List[int], int]

    def __append_zimohu_candidate(
            self, hand: List[int], melds: List[_Meld], zimo_tile: int,
            tool_config: int, candidates: List[int]) -> None:
        assert isinstance(hand, list)
        assert isinstance(melds, list)
        assert isinstance(zimo_tile, int)
        assert 0 <= zimo_tile and zimo_tile < 37
        assert isinstance(tool_config, int)
        assert tool_config >= 0
        assert isinstance(candidates, list)

        melds = [
                Meld(meld_type=meld_type, tiles=tiles, opened=(opened == 1))
                for meld_type, tiles, opened in melds]
        win_tile = self._convert_to_win_tile(zimo_tile)
        hand_config = self._decode_tool_config(tool_config)

        response = self.__hand_calculator.estimate_hand_value(
            tiles=hand, win_tile=win_tile, melds=melds, dora_indicators=[],
            config=hand_config)

        if response.error == 'Hand is not winning': #HandCalculator.ERR_HAND_NOT_WINNING:
            return
        if response.error == 'There are no yaku in the hand': #HandCalculator.ERR_NO_YAKU:
            return
        if response.error is None:
            candidates.append(219)
            return

        raise RuntimeError(response.error)

    def append_zimohu_candidate(
            self, hand: List[int], melds: List[_Meld], zimo_tile: int,
            tool_config: int, candidates: List[int]) -> None:
        try:
            self.__append_zimohu_candidate(
                hand, melds, zimo_tile, tool_config, candidates)
        except Exception:
            print_exc(file=sys.stderr)
            raise

    def __append_rong_candidate(
            self, relseat: int, hand: List[int], melds: List[_Meld],
            zimo_tile: int, tool_config: int, candidates: List[int]) -> None:
        assert isinstance(relseat, int)
        assert 0 <= relseat and relseat < 3
        assert isinstance(hand, list)
        assert len(hand) in (2, 5, 8, 11, 14)
        assert isinstance(melds, list)
        assert isinstance(zimo_tile, int)
        assert 0 <= zimo_tile and zimo_tile < 37
        assert isinstance(tool_config, int)
        assert tool_config >= 0
        assert isinstance(candidates, list)

        melds = [
                Meld(meld_type=meld_type, tiles=tiles, opened=(opened == 1))
                for meld_type, tiles, opened in melds]
        win_tile = self._convert_to_win_tile(zimo_tile)
        hand_config = self._decode_tool_config(tool_config)

        response = self.__hand_calculator.estimate_hand_value(
            tiles=hand, win_tile=win_tile, melds=melds, dora_indicators=[],
            config=hand_config)

        if response.error == 'Hand is not winning': # HandCalculator.ERR_HAND_NOT_WINNING:
            return
        if response.error == 'There are no yaku in the hand': # HandCalculator.ERR_NO_YAKU:
            return
        if response.error is None:
            candidates.append(543 + relseat)
            return

        raise RuntimeError(response.error)

    def append_rong_candidate(
            self, relseat: int, hand: List[int], melds: List[_Meld],
            zimo_tile: int, tool_config: int, candidates: List[int]) -> None:
        try:
            self.__append_rong_candidate(
                    relseat, hand, melds, zimo_tile, tool_config, candidates)
        except Exception:
            print_exc(file=sys.stderr)
            raise

    def __calculate_hand(
            self, hand: List[int], melds: List[_Meld], hupai: int,
            dora_indicators: List[int], tool_config: int) -> Tuple[int, int]:
        assert isinstance(hand, list)
        assert len(hand) in (2, 5, 8, 11, 14)
        assert isinstance(melds, list)
        assert isinstance(hupai, int)
        assert 0 <= hupai and hupai < 37
        assert isinstance(dora_indicators, list)
        assert isinstance(tool_config, int)
        assert tool_config >= 0

        win_tile = self._convert_to_win_tile(hupai)
        melds = [
                Meld(meld_type=meld_type, tiles=tiles, opened=(opened == 1))
                for meld_type, tiles, opened in melds]
        hand_config = self._decode_tool_config(tool_config)

        response = self.__hand_calculator.estimate_hand_value(
                tiles=hand, win_tile=win_tile, melds=melds,
                dora_indicators=dora_indicators, config=hand_config)

        if response.error is None:
            if 13 <= response.han and response.han < 26:
                # 数え役満は13翻に切り揃える．
                # TODO: 26翻以上の数え役満に対応する．
                return (13, response.fu)
            return (response.han, response.fu)

        raise RuntimeError(response.error)


    def calculate_hand(
            self, hand: List[int], melds: List[_Meld], hupai: int,
            dora_indicators: List[int], tool_config: int) -> Tuple[int, int]:
        try:
            return self.__calculate_hand(hand, melds, hupai, dora_indicators, tool_config)
        except Exception:
            print_exc(file=sys.stderr)
            raise
