*"It's time to get me in."* - Kana Ikeda

# kanachan
A Mahjong AI that supports a variant of rules of 4-player Japanese Riichi Mahjong that is adopted in standard games in Mahjong Soul (雀魂, Jantama, [CN](https://www.maj-soul.com/), [JP](https://mahjongsoul.com/), [EN](https://mahjongsoul.yo-star.com/)).

## Brief Guide

This repository provides an annotation tool for game records of Mahjong Soul, and programs training some types of Mahjong AI models. However, this repository does not provide any crawler for game records of Mahjong Soul, any training data, nor any trained models. Therefore, users are assumed to prepare their own training data and computation resources.

The first thing users should do in order to use this repository is to collect game records of Mahjong Soul. The format of game records must be the same as the WebSocket response message that is returned from the Mahjong Soul API server when you hit a URL of the format `https://game.mahjongsoul.com/?paipu=YYMMDD-XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX`. Data in this format can be obtained by capturing WebSocket messages exchanged with the Mahjong Soul's API server, using a network sniffering tools such as [mitmproxy](https://mitmproxy.org/) or [Wireshark](https://www.wireshark.org/), browser extensions, or other tools. Again, this repository does not include such a tool. Therefore, please look for one on code hosting services including GitHub, or implement one yourself.

After collecting game records, the next step is to use [annotate](src/annotation) to convert the game records into annotations in a format suitable for learning.

Finally, the trained model can be obtained by running the training programs under the [kanachan](kanachan) Python module with the annotations as input.

## Goal of This Project

The goal of this project is to create a Mahjong AI for a variant of rules of 4-player Japanese Riichi Mahjong that can beat existing top-tier Mahjong AIs, including [NAGA](https://dmv.nico/ja/articles/mahjong_ai_naga/) and [Suphx](https://arxiv.org/abs/2003.13590), and even top professional human players.

This project is a personal one by myself. This is in contrast to some of the top mahjong AI projects today, which are run by corporations. This project is also intended to show the world that top-class mahjong AI can be built by personal projects.

Currently, Japanese chess (Shogi, 将棋) AI has been already considered to be far superior to the level of top human professionals. I believe that the driving force behind this situation in Japanese chess is the fierce competition among various Shogi AI by personal projects. I expect for this project to be a pioneer to cause such situation in the field of mahjong AI, too.

## Key Features of This Project

### Extremely Large Data Set from Mahjong Soul

This project supposes to use game record (牌譜) data set crawled from Mahjong Soul. This would become an extremely large data set, which differs in order of magnitude in both quantity and generation speed from the existing representative, i.e., the one from the Phoenix Table (houou-taku, 鳳凰卓) of [Tenhou (天鳳)](https://tenhou.net/).

Let me show you concrete numbers. Game records consisting of 17 million rounds, which were generated in 11 years from 2009 to 2019, can be obtained from the Phoenix Table of Tenhou. On the other hand, I have been crawling game records from Mahjong Soul since July 2020, and the amount of game records for 4-player Mahjong played in the Gold Room (kin-no-ma, 金の間) or the higher rooms has reached about 65 million rounds as of the end of August 2021. This number will surely surpass 100 million rounds by the end of 2021.

This critical difference in data volume will allow us to use models that are orders of magnitude larger and/or more powerfully expressive than those used in existing Mahjong AIs. For example, while NAGA and Suphx have trained the [ResNet](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) with the Tenhou dataset, this project aims to take advantage of the huge amount of data to train large scale models based on more powerfully expressive framework (e.g., transformer).

### No Human-crafted Features

The inputs to models in this project, the features in other words, are almost devoid of processing based on human experience and intuition about Mahjong. All tiles are represented as mere tokens, which are indexes to the corresponding embeddings. The token representing the 1 Circle (一筒) tile is not directly associated with the number "1", nor it does indicate one of Circle tiles. The tokens that represent the 1 Circle tile in one's hand and ones that represent the 1 Circle tile in their discarded tiles (河) are not directly related at all. There is no feature that directly represents the relationship between the dora-indicating tile (ドラ表示牌) and the dora (ドラ) tile. There is no feature that represents the visible-to-player number of tiles of a certain type. While there are a total of 90 types of chows (chi, チー, 吃) in the standard rule of Mahjong Soul, each chow is represented by only one of completely independent 90 tokens... and so on.

The situation at a given time in a game is represented very simply as follows. Aspects of game situation that have nothing to do with the order in which the game is played, such as the game wind (chang, 場), the round number (ju, 局), the dora tiles, the hand tiles, etc., are represented as a set of above-mentioned tokens. The discarded tiles (打牌) and the meldings (fulu, 副露) played by each player are represented as a sequence of above-mentioned tokens representing the order in which they occur. The number of points, the number of riichi deposits, and other numerically meaningful features are represented as numbers themselves.

To be more specific, see [Training Data Format for Behavioral Cloning](https://github.com/Cryolite/kanachan/wiki/Notes-on-Training-Data#training-data-format-for-behavioral-cloning).

Some readers may be seriously wondering whether such feature design is really capable of proper learning. Don't worry. Even in the very early stages of learning, the behavior of the model trained with the above-mentioned feature design already shows that it has acquired basic concepts of Mahjong. It seems to acquire concepts including dora, the red tiles, the Dragon Tiles (箭牌), the game wind tile (圏風牌), the player's wind tile (門風牌), 断幺九, melding (鳴き) for fans (役) including 断幺九, 三色同順, 一気通貫, 混全帯么九, and 対々和, value of 混一色 and 清一色, merely formal ready hands (形式聴牌), getting out of the game, 現物 (gen-butsu, the concept that tiles discarded after a riichi is absolutely safe against that riichi), 筋 (suji, the concept that, for example, if 5s is discarded after a riichi, 2s and 8s are relatively safe against that riichi), Liuju Manguan (流し満貫)... and so on.

However, it goes without saying that such an end-to-end feature design requires large data sets and highly expressive models to function properly. It is a fundamental trade-off in machine learning whether to use human wisdom to devise appropriate feature designs, or to prepare large datasets and highly expressive models and leave them to large-scale computational resources. This project chooses the latter, because the essence of the success of deep learning is the liberation from feature engineering, and because I have been engaged in machine learning since the early 2000s and struggled with feature engineering in those days.

### Step-by-step Curriculum Fine-tuning

There are various objectives in Mahjong AIs, including imitation of human behavior, maximization of round delta of the score, higher final ranking, and maximization of delta of the grading point (段位戦ポイント). Since these objectives become more abstract and comprehensive in this order, the latter learning we move to, the more difficult it becomes to learn.

The idea behind this project is to learn mappings from action selections to these objectives step by step, from the easiest to the hardest. This would be equivalent to [Curriculum Learning](https://dl.acm.org/doi/10.1145/1553374.1553380). Moreover, when a mapping for one objective has been learned and then starting learning a mapping for one more harder objective, the *encoder part* of the model trained in the former step is reused in the training of the latter mapping, and only the *decoder part* of the model is replaced to tailor to the new harder objective. The information learned in the former step is stored in the encoder part and transferred to the latter step. By doing so, it is intended that universal knowledge about Mahjong that is independent of objectives will be retained in the encoder part. In this project, this idea is called *curriculum fine-tuning*.

## Components

### [prerequisites](prerequisites) (For developers only)

Make various prerequisite packages and tools available for use in other components. This component is built and available as a [public Docker image](https://hub.docker.com/r/cryolite/kanachan.prerequisites), and implicitly used by other components. Therefore, there is no need for non-developers to build or directly use this component.

### [src/annotation](src/annotation)

A C++ program that extracts almost all the *decision-making points* from game records of Mahjong Soul, and converts the game situation at each decision-making point together with the player's action and round's/game's final results into annotations suitable to learning.

### [src/xiangting](src/xiangting)

A C++ program that generates a [LOUDS-based TRIE data structure](https://github.com/s-yata/marisa-trie) used to calculate shanten (xiang ting, 向聴) numbers.

### [src/simulation](src/simulation)

A C++ library implementing a Mahjong simulator that perfectly mimics the standard game rule of Mahjong Soul, including even many unstated corner cases of the rule. The functionality of this library can be also accessed via the `kanachan.simulation.simulate` Python function.

### [src/paishan](src/paishan)

A C++ program that restores the entire tile wall (pai shan, 牌山) from a game record of Mahjong Soul. Note that the tile wall restored by this program can be used for testing purposes (input to [test/annotation_vs_simulation](test/annotation_vs_simulation)) only, and not for any other purpose.

### [test/annotation_vs_simulation](test/annotation_vs_simulation)

A testing framework called *annotation-vs-simulation* that checks if there is any discrepancy between the annotation implementation and the simulation implementation.

### [kanachan/training](kanachan/training)

Implementations of learning programs and prediction modules with [PyTorch](https://pytorch.org/).
