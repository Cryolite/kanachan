*"It's time to get me in."* - Kana Ikeda

# kanachan
A Mahjong AI that supports a variant of rules of 4-player Japanese Riichi Mahjong that is adopted in standard games in Mahjong Soul (雀魂).

## Goal of This Project

The goal of this project is to create a Mahjong AI for a variant of rules of 4-player Japanese Riichi Mahjong that can beat existing top-tier Mahjong AIs, including [NAGA](https://dmv.nico/ja/articles/mahjong_ai_naga/) and [Suphx](https://arxiv.org/abs/2003.13590), and even top professional human players.

This project is an individual one by myself. This is in contrast to some of the top mahjong AI projects today, which are run by corporations. This project is also intended to show the world that top-class mahjong AI can be built by individual projects.

Currently, Japanese chess (Shogi, 将棋) AI has been already considered to be far superior to the level of top human professionals. I believe that the driving force behind this situation in Japanese chess is the fierce competition among various Shogi AI by individual projects. I expect for this project to be a pioneer to cause such situation in the field of mahjong AI, too.

## Key Features of This Project

### Extremely Large Data Set from Mahjong Soul

This project supposes to use game record (牌譜) data set crawled from [Mahjong Soul (Jantama, 雀魂)](https://mahjongsoul.com/). This would become an extremely large data set, which differs in order of magnitude in both quantity and generation speed from the existing representative, i.e., the one from the Phoenix Table (houou-taku, 鳳凰卓) of [Tenhou (天鳳)](https://tenhou.net/).

Let me show you concrete numbers. Game records consisting of 17 million rounds, which were generated in 11 years from 2009 to 2019, can be obtained from the Phoenix Table of Tenhou. On the other hand, I have been crawling game records from Mahjong Soul since July 2020, and the amount of game records for 4-player Mahjong played in the Gold Room (kin-no-ma, 金の間) or the higher rooms has reached about 65 million rounds as of the end of August 2021. This number will surely surpass 100 million rounds by the end of 2021.

This critical difference in data volume will allow us to use models that are orders of magnitude larger and/or more powerfully expressive than those used in existing Mahjong AIs. For example, while NAGA and Suphx have trained the [ResNet](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) with the Tenhou dataset, this project aims to take advantage of the huge amount of data to train large scale models based on more powerfully expressive framework (e.g., transformer).

### No Human-crafted Features

The inputs to models in this project, the features in other words, are almost devoid of processing based on human experience and intuition about Mahjong. All tiles are represented as mere tokens, which are indexes to the corresponding embeddings. The token representing the 1 Circle (一筒) tile is not directly associated with the number "1", nor it does indicate one of Circle tiles. The tokens that represent the 1 Circle tile in one's hand and ones that represent the 1 Circle tile in their discarded tiles (河) are not directly related at all. There is no feature that directly represents the relationship between the dora-indicating tile (ドラ表示牌) and the dora (ドラ) tile. There is no feature that represents the visible-to-player number of tiles of a certain type. While there are a total of 90 types of chows (chi, チー, 吃) in the standard rule of Mahjong Soul, each chi is represented by only one of completely independent 90 tokens... and so on.

The situation at a given time in a game is represented very simply as follows. Aspects of game situation that have nothing to do with the order in which the game is played, such as the game wind (chang, 場), the round number (ju, 局), the dora tiles, the hand tiles, etc., are represented as a set of above-mentioned tokens. The discarded tiles (打牌) and the meldings (fulu, 副露) played by each player are represented as a sequence of above-mentioned tokens representing the order in which they occur. The number of points, the number of deposits, and other numerically meaningful features are represented as numbers themselves.

To be more specific, see [Annotation Specification](#annotation-specification) section.

Some readers may be seriously wondering whether such feature design is really capable of proper learning. Don't worry. Even in the very early stages of learning, the behavior of the model trained with the above-mentioned feature design already shows that it has acquired basic concepts of Mahjong. It seems to acquire concepts including dora, the red tiles, the Dragon Tiles (箭牌), the game wind tile (圏風牌), the player's wind tile (門風牌), 断幺九, melding (鳴き) for fans (役) including 断幺九, 三色同順, 一気通貫, 混全帯么九, and 対々和, value of 混一色 and 清一色, merely formal ready hands (形式聴牌), getting out of the game, 現物 (gen-butsu, the concept that tiles discarded after a riichi is absolutely safe against that riichi), 筋 (suji, the concept that, for example, if 5s is discarded after a riichi, 2s and 8s are relatively safe against that riichi), Liuju Manguan (流し満貫)... and so on.

However, it goes without saying that such an end-to-end feature design requires large data sets and highly expressive models to function properly. It is a fundamental trade-off in machine learning whether to use human wisdom to devise appropriate feature designs, or to prepare large datasets and highly expressive models and leave them to large-scale computational resources. This project chooses the latter, because the essence of the success of deep learning is the liberation from feature engineering, and because I have been engaged in machine learning since the early 2000s and had struggled with feature engineering in those days.

### Step-by-step Curriculum Fine-tuning

There are various objectives in Mahjong AIs, include imitation of human behavior, maximization of round delta of the score, higher final ranking, and maximization of delta of the grading point (段位戦ポイント). Since these objectives become more abstract and comprehensive in this order, the latter learning we move to, the more difficult it becomes to learn.

The idea behind this project is to learn mappings from action selections to these objectives step by step, from the easiest to the hardest. This would be equivalent to [Curriculum Learning](https://dl.acm.org/doi/10.1145/1553374.1553380). Moreover, when a mapping for one objective has been learned and then start learning a mapping for one more harder objective, the *encoder part* of the model trained in the former step is reused in the training of the latter mapping, and only the *decoder part* of the model is replaced to tailor to the new harder objective. The information learned in the former step is stored in the encoder part and transferred to the latter step. By doing so, it is intended that universal knowledge about Mahjong that is independent of objectives will be retained in the encoder part. In this project, this idea is called *curriculum fine-tuning*.

## Components

### [prerequisites](prerequisites) (For developers only)

Make various prerequisite packages and tools available for use in [annotate](#annotate). This component is built and available as a [public Docker image](https://hub.docker.com/r/cryolite/kanachan.prerequisites), and implicitly used by annotate. Therefore, there is no need for non-developers to build or directly use this component.

### [annotate](annotate)

A C++ program that extracts all the *decision-making points* from game records of Mahjong Soul, and converts the game situation at each decision-making point together with the player's action and round's/game's final results into annotations. The output of the program is text, where each line represents the annotation of a decision-making point. The format of the annotation (each line of the output) is specified in the [Annotation Specification](#annotation-specification) section.

A *decision-making point* is defined by a point in a game at which a player is forced to choose an action among multiple options. Possible points and actions are enumerated as follows:

* Immediately after a self-draw (zimo, 自摸):
  * which tile in their hand to discard if not in riichi,
  * whether or not to declare riichi if possible,
  * whether or not to declare to win (zimo hu, 自摸和) if possible,
  * whether or not to declare a closed kong (an gang, 暗槓) if possible,
  * whether or not to declare a open kong (jia gang, 加槓) if possible, and
  * whether or not to declare no game (jiu zhong jiu pai, 九種九牌) if possible.
* Immediately after other player's discard:
  * whether or not to declare chi if possible,
  * whether or not to declare pon (peng, ポン, 碰) if possible,
  * whether or not to declare kong (da ming gang, 大明槓) if possible, and
  * whether or not to declare to win (rong, 栄和) if possible.
* Immediately after other player's kong:
  * whether or not declare to win (qiang gang, 槍槓) if possible.

### [kanachan](kanachan)

Implementations of learning programs and prediction modules with [PyTorch](https://pytorch.org/).

## Annotation Specification

The annotation of a decision-making point is represented by one text line. Each line is tab-separated into 7 fields, and each field is in turn comma-separated into elements.

### Common Conventions

Before explaining the details of each field in an annotation, the following explains the conventions used in annotations.

#### Seat

Each player, of course there are four players in a 4-player mahjong game, is distinguished by the notion of "seat"; the 0th seat is the the dealer (zhuang jia, 荘家) of the start of a game (qi jia, 起家), the 1st seat the right next to the 0th seat (xia jia of qi jia, 起家の下家), the 2nd seat the one across from the 0th seat (dui mian of qi jia, 起家の対面), and the 3rd seat the left next to the 0th seat (shang jia of qi jia, 起家の上家).

| Seat | Meaning                           |
|------|-----------------------------------|
| `0`  | the dealer of the start of a game |
| `1`  | the right next to the 0th seat    |
| `2`  | the one across from the 0th seat  |
| `3`  | the left next to the 0th seat     |

#### Relative Seat (Relseat)

There are cases where the relative positions of two players need to be represented. For example, complete information about a pon includes information about who melds the pon and who discards the melded tile. In such a case, one information is represented by a seat index, and the other information is represented by the position relative to the former.

| Relseat | Meaning                                         |
|---------|-------------------------------------------------|
| `0`     | the player right next to the player of interest |
| `1`     | the player across from the player of interest   |
| `2`     | the player left next to the player of interest  |

#### Tile

The type of a tile is represented by an integer from 0 to 36, inclusive.

| Tile    | Value       |
|---------|-------------|
| 0m ~ 9m | `0` ~ `9`   |
| 0p ~ 9p | `10` ~ `19` |
| 0s ~ 9s | `20` ~ `29` |
| 1z ~ 7z | `30` ~ `36` |

#### Tile'

There is no need to distinguish between black and red tiles of certain kinds to indicate a type of closed kong. In such a case, the 34 types of tiles excluding red ones are represented by integers from 0 to 33, inclusive.

| Tile    | Value       |
|---------|-------------|
| 1m ~ 9m | `0` ~ `8`   |
| 1p ~ 9p | `9` ~ `17`  |
| 1s ~ 9s | `18` ~ `26` |
| 1z ~ 7z | `27` ~ `33` |

#### Grade

The grade (段位) is represented by integers from 0 to 15, inclusive.

| Grade   | Value       |
|---------|-------------|
| 初心1~3 | `0` ~ `2`   |
| 雀士1~3 | `3` ~ `5`   |
| 雀傑1~3 | `6` ~ `8`   |
| 雀豪1~3 | `9` ~ `11`  |
| 雀聖1~3 | `12` ~ `14` |
| 魂天    | `15`        |

#### Chow (Chi)

Chows are represented by integers from 0 to 89, inclusive.

| Value       | Chow (The last element represents the discarded tile) |
|-------------|-------------------------------------------------------|
| `0`         | (2m, 3m, 1m)                                          |
| `1`         | (1m, 3m, 2m)                                          |
| `2`         | (3m, 4m, 2m)                                          |
| `3`         | (1m, 2m, 3m)                                          |
| `4`         | (2m, 4m, 3m)                                          |
| `5`         | (4m, 5m, 3m)                                          |
| `6`         | (4m, 0m, 3m)                                          |
| `7`         | (2m, 3m, 4m)                                          |
| `8`         | (3m, 5m, 4m)                                          |
| `9`         | (3m, 0m, 4m)                                          |
| `10`        | (5m, 6m, 4m)                                          |
| `11`        | (0m, 6m, 4m)                                          |
| `12`        | (3m, 4m, 5m)                                          |
| `13`        | (3m, 4m, 0m)                                          |
| `14`        | (4m, 6m, 5m)                                          |
| `15`        | (4m, 6m, 0m)                                          |
| `16`        | (6m, 7m, 5m)                                          |
| `17`        | (6m, 7m, 0m)                                          |
| `18`        | (4m, 5m, 6m)                                          |
| `19`        | (4m, 0m, 6m)                                          |
| `20`        | (5m, 7m, 6m)                                          |
| `21`        | (0m, 7m, 6m)                                          |
| `22`        | (7m, 8m, 6m)                                          |
| `23`        | (5m, 6m, 7m)                                          |
| `24`        | (0m, 6m, 7m)                                          |
| `25`        | (6m, 8m, 7m)                                          |
| `26`        | (8m, 9m, 7m)                                          |
| `27`        | (6m, 7m, 8m)                                          |
| `28`        | (7m, 9m, 8m)                                          |
| `29`        | (7m, 8m, 9m)                                          |
| `30` ~ `59` | Likewise for Circle tiles (筒子)                      |
| `60` ~ `89` | Likewise for Bamboo tiles (索子)                      |

### 0th Field: Game UUID

The 0th field is the game UUID, which uniquely identifies the game in which the decision-making point appears. This field is for debugging purposes only and is not used for learning at all.

### 1st Field: Sparse Features

The 1st field consists of *sparse features*. All the elements in this field are an non-negative integer. These integers are used as indices for embeddings, which are finally used as a part of inputs to learning models. The meaning of each integer is as follows.

| Title                              | Value                                                | Note                        |
|------------------------------------|------------------------------------------------------|-----------------------------|
| Room                               | `0`: Bronze Room (銅の間)<br/>`1`: Silver Room (銀の間)<br/>`2`: Gold Room (金の間)<br/>`3`: Jade Room (玉の間)<br/>`4`: Throne Room (王座の間) ||
| Game Style                         | `5`: quarter-length game (dong feng zhan, 東風戦)<br/>`6`: half-length game (ban zhuang zhan, 半荘戦) ||
| Seat                               | `7` ~ `10`                                           | `7 + seat`                  |
| Game Wind (Chang, 場)              | `11`: East (東場)<br/>`12`: South (南場)<br/>`13`: West (西場) |                   |
| Round (Ju, 局)                     | `14` ~ `17`                                          | `14 + round`                |
| Dora Indicator                     | `18` ~ `54`                                          | `18 + tile`                 |
| 2nd Dora Indicator                 | `55` ~ `91`                                          | optional, `55 + tile`       |
| 3rd Dora Indicator                 | `92` ~ `128`                                         | optional, `92 + tile`       |
| 4th Dora Indicator                 | `129` ~ `165`                                        | optional, `129 + tile`      |
| 5th Dora Indicator                 | `166` ~ `202`                                        | optional, `166 + tile`      |
| # of Left Tiles to Draw            | `203` ~ `272`                                        | `# of left tiles = 272 - x` |
| Grade of 0th Seat (起家段位)       | `273` ~ `288`                                        | `273 + grade`               |
| Rank of 0th Seat (起家順位)        | `289` ~ `292`                                        | `289 + rank`                |
| Grade of 1st Seat (起家の下家段位) | `293` ~ `308`                                        | `293 + grade`               |
| Rank of 1st Seat (起家の下家順位)  | `309` ~ `312`                                        | `309 + rank`                |
| Grade of 2nd Seat (起家の対面段位) | `313` ~ `328`                                        | `313 + grade`               |
| Rank of 2nd Seat (起家の対面順位)  | `329` ~ `332`                                        | `329 + rank`                |
| Grade of 3rd Seat (起家の上家段位) | `333` ~ `348`                                        | `333 + grade`               |
| Rank of 3rd Seat (起家の上家順位)  | `349` ~ `352`                                        | `349 + rank`                |
| Hand (shou pai, 手牌)              | `353` ~ `488`                                        | combination of tiles        |
| Drawn Tile (zimo pai, 自摸牌)      | `489` ~ `525`                                        | optional, `489 + tile`      |

### 2nd Field: Numeric Features

The 2nd field consists of *numeric features*. This field consists of exactly 6 elements. These features are numerically meaningful and directly used as a part of inputs to learning models. The meaning of each element is as follows.

| Element Index  | Explanation                                    |
|----------------|------------------------------------------------|
| 0              | The number of counter sticks (ben chang, 本場) |
| 1              | The number of riichi deposits (供託本数)       |
| 2              | The score of the 0th seat                      |
| 3              | The score of the 1st seat                      |
| 4              | The score of the 2nd seat                      |
| 5              | The score of the 3rd seat                      |

### 3rd Field: Progression Features

The 3rd field consists of *progression features*. This field represents a sequence of non-negative integers. Each integer stands for some event in a round of a game. The order of the integers in the sequence directly represents the order in which the events occur in the round of the game. These integers are used as indices for embeddings, which are finally used as a part of inputs to learning models. Note, however, that positional encoding must be applied to the embeddings if they are to be used as a part of inputs to models such as ones using transformer, which erase the positional/order information of the input embeddings. The meaning of each integer is as follows.

| Title                  | Values          | Note                                      |
|------------------------|-----------------|-------------------------------------------|
| Begging of Round       | `0`             | Always starts with this feature           |
| Discard of Tile (打牌) | `5` ~ `596`     | `5 + seat * 148 + tile * 4 + a * 2 + b`, where;<br/>`a = 0`: not moqi (手出し)<br/>`a = 1`: moqi (自摸切り)<br/>`b = 0`: w/o riichi declaration<br/>`b = 1`: w/ riichi declaration |
| Chow (Chi, チー)       | `597` ~ `956`   | `597 + seat * 90 + chi`                   |
| Pon (peng, ポン)       | `957` ~ `1400`  | `957 + seat * 111 + relseat * 37 + tile`  |
| Da Ming Gang (大明槓)  | `1401` ~ `1844` | `1401 + seat * 111 + relseat * 37 + tile` |
| An Gang (暗槓)         | `1845` ~ `1980` | `1845 + seat * 34 + tile'`                |
| Jia Gang (加槓)        | `1981` ~ `2128` | `1981 + seat * 37 + tile`                 |

### 4th Field: Possible Actions

The 4th field consists of all the possible actions at that decision-making point.

| Type of Actions              | Value         | Note                                                                      |
|------------------------------|---------------|---------------------------------------------------------------------------|
| Discarding tile              | `0` ~ `147`   | `tile * 4 + a * 2 + b`, where;<br/>`a = 0`: not moqi (手出し)<br/>`a = 1`: moqi (自摸切り)<br/>`b = 0`: w/o riichi declaration<br/>`b = 1`: w/ riichi declaration |
| An Gang (暗槓)               | `148` ~ `181` | `148 + tile'`                                                             | 
| Jia Gang (加槓)              | `182` ~ `218` | Represented by the tile newly added to an existing peng.<br/>`182 + tile` |
| Zimo Hu (自摸和)             | `219`         |                                                                           |
| Jiu Zhong Jiu Pai (九種九牌) | `220`         |                                                                           |
| Skip                         | `221`         |                                                                           |
| Chi (チー, 吃)               | `222` ~ `311` | `222 + chi`                                                               |
| Peng (ポン)                  | `312` ~ `422` | `312 + relseat * 37 + tile`                                               |
| Da Ming Gang                 | `423` ~ `533` | Represented by the discarded tile.<br/>`423 + relseat * 37 + tile`        |
| Rong (栄和)                  | `534` ~ `536` | `534`: from xia jia (下家から)<br/>`535`: from dui mian (対面から)<br/>`536`: from shang Jia (上家から) |

### 5th Field: Actual Action

The 5th field indicates the actual action chosen by the player at that decision-making point. This field is the index to one of the possible actions enumerated in the 4th field.

### 6th Field: Results

The 6th field consists of some aspects of the final result of the round and game in which the decision-making point appear. This field consists of exactly 12 elements.

| Element Index  | Explanation                                                                    |
|----------------|--------------------------------------------------------------------------------|
| 0              | End-of-round result from the point of view of the player indicated by **Seat**<br/>`0`: 自家自摸和<br/>`1`: 下家自摸和<br/>`2`: 対面自摸和<br/>`3`: 上家自摸和<br/>`4`: 下家からの自家栄和<br/>`5`: 対面からの自家栄和<br/>`6`: 上家からの自家栄和<br/>`7`: 下家への放銃<br/>`8`: 対面への放銃<br/>`9`: 上家への放銃<br/>`10`: 下家へ対面から横移動<br/>`11`: 下家へ上家から横移動<br/>`12`: 対面へ下家から横移動<br/>`13`: 対面へ上家から横移動<br/>`14`: 上家へ下家から横移動<br/>`15`: 上家へ対面から横移動<br/>`16`: 荒牌平局 (不聴)<br/>`17`: 荒牌平局 (聴牌)<br/>`18`: 途中流局|
| 1              | Round delta of the score of the 0th seat                                       |
| 2              | Round delta of the score of the 1st seat                                       |
| 3              | Round delta of the score of the 2nd seat                                       |
| 4              | Round delta of the score of the 3rd seat                                       |
| 5              | End-of-game ranking of the 0th seat                                            |
| 6              | End-of-game ranking of the 1st seat                                            |
| 7              | End-of-game ranking of the 2nd seat                                            |
| 8              | End-of-game ranking of the 3rd seat                                            |
| 9              | End-of-game ranking of the player indicated by **Seat**                        |
| 10             | End-of-game score of the player indicated by **Seat**                          |
| 11             | Game delta of grading score of the player indicated by **Seat**                |
